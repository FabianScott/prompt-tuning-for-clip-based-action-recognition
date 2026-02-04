import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from .CoOp import CoOpModel
from ..configs.parse_run_args import load_model_from_keys
from ..configs.paths import EVAL_PATH
from ..plots import plot_reliability_diagram
from ..eval.eval_utils import (
    setup_evaluation_environment,
    generate_test_path,
    load_config_and_checkpoint,
    prepare_method_config,
    build_evaluation_dataset,
    build_evaluation_dataloader,
    init_wandb_logger
)

def expected_calibration_error(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.any():
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            ece += (mask.float().mean()) * torch.abs(acc - conf)

    return ece.item()

def brier_score(probs, labels, num_classes):
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return torch.mean(torch.sum((probs - one_hot) ** 2, dim=1)).item()


def maximum_calibration_error(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_edges = torch.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.any():
            acc = accuracies[mask].float().mean()
            conf = confidences[mask].mean()
            mce = max(mce, torch.abs(acc - conf).item())

    return mce


def adaptive_ece(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    sorted_idx = torch.argsort(confidences)
    confidences = confidences[sorted_idx]
    accuracies = accuracies[sorted_idx]

    bin_size = len(confidences) // n_bins
    ece = 0.0

    for i in range(n_bins):
        start = i * bin_size
        end = len(confidences) if i == n_bins - 1 else (i + 1) * bin_size
        conf_bin = confidences[start:end]
        acc_bin = accuracies[start:end]
        if len(conf_bin) > 0:
            ece += (len(conf_bin) / len(confidences)) * torch.abs(
                acc_bin.mean() - conf_bin.mean()
            )

    return ece.item()


def classwise_ece(probs, labels, n_bins=15):
    num_classes = probs.size(1)
    eces = []

    for c in range(num_classes):
        class_probs = probs[:, c]
        class_labels = (labels == c).long()

        bin_edges = torch.linspace(0, 1, n_bins + 1)
        ece_c = 0.0

        for i in range(n_bins):
            mask = (class_probs > bin_edges[i]) & (class_probs <= bin_edges[i + 1])
            if mask.any():
                acc = class_labels[mask].float().mean()
                conf = class_probs[mask].mean()
                ece_c += (mask.float().mean()) * torch.abs(acc - conf)

        eces.append(ece_c)

    return torch.mean(torch.stack(eces)).item()


def run_calibration_evaluation(
    eval_config,
    meta_config,
    model_key: list[str],
    ):
    test_name, _ = generate_test_path(eval_config, prefix="calibration")
    
    setup_evaluation_environment()
    
    # Load config
    method_config, _ = load_config_and_checkpoint(eval_config, model_key=model_key)
    method_config = prepare_method_config(method_config, eval_config)
    
    # Build dataset and dataloader
    dataset, class_names = build_evaluation_dataset(eval_config, method_config)
    dataloader = build_evaluation_dataloader(dataset, meta_config)
    
    # Initialize wandb
    wandb_logger = init_wandb_logger(meta_config, method_config, "Calibration-" + test_name, ["calibration"])
    
    # Load model
    model = load_model_from_keys(
        keys=model_key,
        class_names=class_names,
        wandb_logger=wandb_logger,
    )

    calibrated_model = CalibratedModel(
        model=model,
        temperature=meta_config.get("temperature", 1.0),
        wandb_logger=wandb_logger,
    )

    metrics, all_logits, all_labels = calibrated_model.eval(
        dataloader=dataloader,
        class_names=class_names,
        use_hand_crafted_text_features=eval_config.get("use-handcrafted-prompts", False),
        n_bins=meta_config.get("n-bins", 15),
    )

    plot_reliability_diagram(
        probs=F.softmax(all_logits, dim=1),
        labels=all_labels,
        n_bins=meta_config.get("n-bins", 15),
        save_path=os.path.join(EVAL_PATH, test_name, "reliability_diagram.png"),
        show=False,
    )
    if wandb_logger:
        wandb_logger.finish()


class CalibratedModel(CoOpModel):
    def __init__(self, model: CoOpModel, temperature: float = 1.0, wandb_logger=None):
        super().__init__(
            config=model.config,
            class_names=model.class_names,
            device=model.device,
            eval_class_names=model.eval_class_names,
            wandb_logger=wandb_logger,
        )
        self.model = model
        self.temperature = torch.tensor(temperature, device=self.device)
        self.model.eval_mode()

    @torch.no_grad()
    def eval(
        self,
        dataloader: DataLoader,
        class_names: Optional[list[str]] = None,
        use_hand_crafted_text_features: bool = False,
        n_bins: int = 15,
    ):
        if class_names is None:
            class_names = self.class_names

        all_logits, all_labels = [], []

        if use_hand_crafted_text_features:
            text_features = self.context_learner.get_hand_crafted_features(self.clip).detach()
        else:
            text_features = self.model.encode_texts(class_names=class_names)

        pbar = tqdm(dataloader, desc="Calibration Eval")

        for batch in pbar:
            videos, masks, labels = batch
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device) if masks is not None else None

            logits, _, _ = self.model.forward(
                videos=videos,
                text_features=text_features,
                video_masks=masks,
                class_names=class_names,
                softmax=False,
            )

            logits = logits / self.temperature
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Update progress bar
            probs = F.softmax(logits, dim=1)
            acc = (probs.argmax(dim=1) == labels).float().mean().item()
            pbar.set_postfix({"Batch Acc": f"{acc*100:.2f}%"})

        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)

        probs = F.softmax(logits_cat, dim=1)

        metrics = {
            "nll": F.nll_loss(torch.log(probs), labels_cat).item(),
            "ece": expected_calibration_error(probs, labels_cat, n_bins),
            "accuracy": (probs.argmax(dim=1) == labels_cat).float().mean().item(),
            "brier_score": brier_score(probs, labels_cat, num_classes=len(class_names)),
            "mce": maximum_calibration_error(probs, labels_cat, n_bins),
            "adaptive_ece": adaptive_ece(probs, labels_cat, n_bins),
            "classwise_ece": classwise_ece(probs, labels_cat, n_bins)
        }

        if self.wandb_logger is not None:
            self.wandb_logger.log(metrics)

        return metrics, logits_cat, labels_cat
