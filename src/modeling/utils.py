import wandb
import torch
import cProfile
import pstats
import atexit

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from torchmetrics.classification import ConfusionMatrix, Precision, Recall, F1Score, Accuracy


def regularisation_loss(hand_crafted_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    # Compute the cosine similarity
    similarity = torch.cosine_similarity(hand_crafted_features, text_features)
    # Compute the regularisation loss (mean negative similarity)
    loss = -similarity.mean()
    return loss


def mean_pooling(vid_feats: torch.Tensor, mask=None, dim=2):
    if mask is not None:
        lengths = mask.sum(dim=dim).clamp(min=1)[..., None]  # avoid division by zero
        vid_feats = vid_feats.sum(dim=dim) / lengths
    else:
        vid_feats = vid_feats.mean(dim=dim)
    return vid_feats

def max_pooling(vid_feats, mask=None, dim=2):
    if mask is not None:
        # Set masked positions to a very low value before max
        vid_feats = vid_feats.masked_fill(~mask[..., None], float('-inf'))
        vid_feats, _ = vid_feats.max(dim=dim)
        # Replace -inf with zeros (if all values were masked)
        vid_feats[vid_feats == float('-inf')] = 0
    else:
        vid_feats, _ = vid_feats.max(dim=dim)
    return vid_feats

class AttentionPooling(torch.nn.Module):
    """
    Applies attention-based pooling over the videos
    [B, V, T, Vid Dim] -> [B, V, Vid Dim]
    """
    def __init__(self, embed_dim, num_heads=1):
        super(AttentionPooling, self).__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.query = torch.nn.Parameter(torch.randn(1, 1, embed_dim))  # Learnable query vector

    def forward(self, vid_feats: torch.Tensor, mask=None, dim=2):
        B, V, T, D = vid_feats.shape
        vid_feats = vid_feats.view(B * V, T, D)  # [B*V, T, D]
        query = self.query.expand(B * V, -1, -1)  # [B*V, 1, D]

        if mask is not None:
            mask = mask.view(B * V, T)  # [B*V, T]
            attn_output, _ = self.attention(query, vid_feats, vid_feats, key_padding_mask=~mask)
        else:
            attn_output, _ = self.attention(query, vid_feats, vid_feats)

        attn_output = attn_output.view(B, V, D)  # [B, V, D]
        return attn_output

def get_temporal_pool_method(name):
    if name == "mean":
        return mean_pooling
    elif name == "max":
        return max_pooling
    else:
        raise ValueError(f"Unknown temporal pooling method: {name}")

def get_pooled_output(last_hidden_state, attention_mask=None):
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        attention_mask.to(dtype=torch.int, device=last_hidden_state.device),
    ] # see https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/clip/modeling_clip.py#L579
    return pooled_output


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        labels_one_hot = F.one_hot(label, num_classes=probs1.size(1)).float()
        probs2 = F.softmax(labels_one_hot)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

def cosine_similarity_loss(c, c_clip):
    """
    Lcos = 1/Nc * Σ (c · c_clip) / (||c|| * ||c_clip||)
    """
    cos_sim = F.cosine_similarity(c, c_clip, dim=-1)
    return 1 - cos_sim.mean()  # minimizing distance (1 - similarity)


def calculate_class_metrics(logits_cat, labels_cat, class_names, logger = None, loss: Optional[float] = None):
    num_classes = len(class_names)
    
    preds_cat = logits_cat.argmax(dim=1)

    cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)(preds_cat, labels_cat).cpu().tolist()

    # Macro & Micro metrics
    prec_macro = Precision(task="multiclass", num_classes=num_classes, average="macro")(preds_cat, labels_cat).item()
    rec_macro  = Recall(task="multiclass", num_classes=num_classes, average="macro")(preds_cat, labels_cat).item()
    f1_macro   = F1Score(task="multiclass", num_classes=num_classes, average="macro")(preds_cat, labels_cat).item()
    acc_macro  = Accuracy(task="multiclass", num_classes=num_classes, average="macro")(preds_cat, labels_cat).item()
    top5_macro = Accuracy(task="multiclass", num_classes=num_classes, top_k=5)(logits_cat, labels_cat).item() if num_classes >=5 else 0.0

    prec_micro = Precision(task="multiclass", num_classes=num_classes, average="micro")(preds_cat, labels_cat).item()
    rec_micro  = Recall(task="multiclass", num_classes=num_classes, average="micro")(preds_cat, labels_cat).item()
    f1_micro   = F1Score(task="multiclass", num_classes=num_classes, average="micro")(preds_cat, labels_cat).item()
    acc_micro  = Accuracy(task="multiclass", num_classes=num_classes, average="micro")(preds_cat, labels_cat).item()

    # Per-class F1 for percentiles
    f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)(preds_cat, labels_cat).cpu().numpy()
    percentiles = {p: float(np.percentile(f1_per_class, p)) for p in [5, 25, 50, 75, 95]}

    metrics = {}
    for name, metric_cls in [
        ('f1', F1Score),
        ('precision', Precision),
        ('recall', Recall),
        ('accuracy', Accuracy)
    ]:
        m = metric_cls(task="multiclass", num_classes=num_classes, average=None)
        metrics[name] = m(preds_cat, labels_cat).cpu().numpy()

    scores_per_class = {
        cls: {
            'f1': metrics['f1'][i],
            'precision': metrics['precision'][i],
            'recall': metrics['recall'][i],
            'accuracy': metrics['accuracy'][i]
        } for i, cls in enumerate(class_names)}

    if logger is not None:
        logger.log({
            "confusion_matrix_raw": {"classes": class_names, "matrix": cm},
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "accuracy_macro": acc_macro,
            "top5_accuracy_macro": top5_macro,
            "precision_micro": prec_micro,
            "recall_micro": rec_micro,
            "f1_micro": f1_micro,
            "accuracy_micro": acc_micro,
            "loss": loss,
            **{f"f1_percentile_{p}": v for p, v in percentiles.items()},
            "scores_per_class": scores_per_class,
        })

    return scores_per_class





def start_profiling():
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler

def stop_profiling(profiler: cProfile.Profile, max_print: int = 75):
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(max_print)

def enable_profiling():
    """Allows for profiling a script even when terminated prematurely."""
    profiler = start_profiling()
    atexit.register(stop_profiling, profiler)

def check_tensor(t: torch.Tensor, name: str, layer_idx: int = -1):
    """
    Checks a tensor and its gradients for NaN or Inf values and prints statistics.
    Raises a RuntimeError if any NaN or Inf values are found.
    """
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"[NaN/Inf] {name} at layer {layer_idx}: mean={t.mean().item():.3e}, std={t.std().item():.3e}, "
            f"min={t.min().item():.3e}, max={t.max().item():.3e}")
        idx = torch.isnan(t).nonzero(as_tuple=False)
        print("First bad index:", idx[0] if len(idx) else None)
        raise RuntimeError(f"NaN/Inf detected in {name}")
    # Check the gradients if present
    if t.grad is not None:
        if torch.isnan(t.grad).any() or torch.isinf(t.grad).any():
            print(f"[NaN/Inf Grad] {name} at layer {layer_idx}: mean={t.grad.mean().item():.3e}, std={t.grad.std().item():.3e}, "
                f"min={t.grad.min().item():.3e}, max={t.grad.max().item():.3e}")
            idx = torch.isnan(t.grad).nonzero(as_tuple=False)
            print("First bad grad index:", idx[0] if len(idx) else None)
            raise RuntimeError(f"NaN/Inf detected in gradient of {name}")
        else:
            print(f"[Grad OK] {name} at layer {layer_idx}: mean={t.grad.mean().item():.3e}, std={t.grad.std().item():.3e}, "
                f"min={t.grad.min().item():.3e}, max={t.grad.max().item():.3e}")
    print(f"[OK] {name} at layer {layer_idx}: mean={t.mean().item():.3e}, std={t.std().item():.3e}, "
        f"min={t.min().item():.3e}, max={t.max().item():.3e}")

class AnomalyLoss:
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        self.class_weights = class_weights
    
    def __call__(self, logits, labels, margin):
    # logits: [B, K] normal-class logits
    # labels: [B] 0<normal, 0=abnormal, to allow for more than one normal label

        normal_conf = F.softmax(logits, dim=1).max(dim=1).values

        loss_normal = (1 - normal_conf)[labels > 0]
        loss_abnormal = F.relu(normal_conf - margin)[labels == 0]

        if self.class_weights is not None:
            loss_normal = loss_normal * self.class_weights[1]
            loss_abnormal = loss_abnormal * self.class_weights[0]

        return torch.cat([loss_normal, loss_abnormal]).mean()

class AnomalyAcc:
    def __init__(self):
        pass
    
    def __init__(self, margin: float):
        self.margin = margin
        if margin is None:
            raise ValueError("Margin must be provided for AnomalyAcc.")

    def __call__(self, logits, labels):
        # Prection is the normal class confidence, 0 is abnormal
        normal_conf = F.softmax(logits, dim=1).max(dim=1).values
        binary_preds = (normal_conf >= self.margin).long()
        binary_labels = (labels > 0).long()
        
        correct = (binary_preds == binary_labels).sum()
        total = labels.size(0)
        return correct, total

class RegularAcc:
    """Returns the number of correct predictions and total"""
    def __init__(self, use_onehot: bool = False):
        self.use_onehot = use_onehot
    
    def __call__(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        labels = labels if not self.use_onehot else torch.argmax(labels, dim=1)
        correct = (preds == labels).sum()
        total = labels.size(0)
        return correct, total

class OneHotCrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None):
        super(OneHotCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        targets: [B, C] one-hot encoded
        """
        log_probs = F.log_softmax(logits, dim=1)
        if self.weight is not None:
            loss = - (targets * log_probs * self.weight).sum(dim=1).mean()
        else:
            loss = - (targets * log_probs).sum(dim=1).mean()
        return loss

def OneHotKLDivLoss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, C]
    targets: [B, C] one-hot encoded
    """
    log_probs = F.log_softmax(logits, dim=1)
    probs_targets = F.softmax(targets, dim=1)
    loss = F.kl_div(log_probs, probs_targets, reduction='batchmean')
    return loss

def get_loss_function(
        loss_name: str, 
        class_weights: Optional[torch.Tensor] = None, 
        margin: Optional[float] = None,
        use_onehot: bool = False
        ) -> callable:
    """Returns a loss function based on the given name."""
    loss_name = loss_name.lower()
    if loss_name == "cross-entropy":
        return OneHotCrossEntropyLoss(weight=class_weights) if use_onehot else nn.CrossEntropyLoss(weight=class_weights) 
    elif loss_name == "kl-divergence":
        return OneHotKLDivLoss if use_onehot else nn.KLDivLoss(reduction="batchmean", )
    elif loss_name == "cosine-similarity":
        # Custom cosine similarity loss
        return cosine_similarity_loss
    elif loss_name == "anomaly":
        if margin is None:
            raise ValueError("Margin must be provided for anomaly loss.")
        return AnomalyLoss(class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    

def get_acc_function(
        is_anomaly: bool = False, 
        margin: Optional[float] = None,
        use_onehot: bool = False,
        ) -> callable:
    """Returns an accuracy function based on whether it's an anomaly detection task."""
    if is_anomaly:
        return AnomalyAcc(margin=margin)
    else:
        return RegularAcc(use_onehot=use_onehot)