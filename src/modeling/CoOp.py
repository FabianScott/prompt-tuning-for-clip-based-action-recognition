import os
import wandb
import torch
import numpy as np
import torch.nn as nn

from .ContextLearners import TextContextLearner
from ..configs import save_config
from ..configs.paths import CLIP_MODEL_CACHE_DIR
from .utils import get_acc_function, regularisation_loss, KLLoss, cosine_similarity_loss, calculate_class_metrics, get_loss_function

from tqdm import tqdm
from typing import Optional
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.classification import ConfusionMatrix, Precision, Recall, F1Score, Accuracy




def gpu_memory_usage():
    gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
    return gpu_mem_gb  # in GB


class CoOpModel:
    def __init__(
            self, 
            config: dict, 
            class_names: list[str],  
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu', 
            eval_class_names: Optional[list[str]] = None,
            wandb_logger = None
            ):
        self.config = config
        self.device = device
        self.class_names = class_names
        self.eval_class_names = eval_class_names # Allowed to be none as the eval function defaults to training classes
        self.num_classes = len(class_names)
        self.model_name = config["clip-model"]
        self.regularisation_strength = config["regularisation-strength"]
        self.cosine_regularisation = config["cosine-regularisation-strength"]
        self.batches_per_backprop = config["dataset-config"]["batches-per-backprop"]
        self.wandb_logger: wandb.WandbLogger = wandb_logger
        self.scaler = torch.amp.GradScaler(
            enabled=config["fp16"],
            init_scale=self.config["grad-scaler-config"]["init-scale"],
            growth_factor=self.config["grad-scaler-config"]["growth-factor"],
            backoff_factor=self.config["grad-scaler-config"]["backoff-factor"],
            growth_interval=self.config["grad-scaler-config"]["growth-interval"],
            device=self.device
            )
        self.class_weights = torch.tensor(config["dataset-config"]["class-weights"]).to(device) if config["dataset-config"]["class-weights"] is not None else None
        # Load CLIP model and tokenizer/processor
        self.clip = CLIPModel.from_pretrained(self.model_name, use_safetensors=True, force_download=False, cache_dir=CLIP_MODEL_CACHE_DIR).to(device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name, force_download=False, use_fast=True, cache_dir=CLIP_MODEL_CACHE_DIR)
        self.tokenizer = self.processor.tokenizer
        self.margin = config["dataset-config"].get("margin-factor", 1.0) / config["dataset-config"].get("num-model-outputs-for-normal", 1) if config["dataset"] == "nwpu" else None
        self.loss_fn = get_loss_function(
            loss_name=config["loss-function"], 
            class_weights=self.class_weights, 
            margin=self.margin,
            use_onehot=config["videomix-type"] is not None
            )
        self.acc_fn = get_acc_function(
            is_anomaly=config["dataset"] == "nwpu", 
            margin=self.margin,
            use_onehot=config["videomix-type"] is not None
            )

        self.use_checkpoint = config["use-checkpointing"]
        self.epochs_to_skip = config["epochs-to-skip"]
        self.grad_norm_max = config["grad-norm-max"]
        self.debug = config["debug"]
        self.early_stop_patience = config["early-stop-patience"]
        self.use_handcrafted_features = config["use-handcrafted-features"]
        self.min_interval_tqdm = 1.0
        # Freeze CLIP weights (vision + text) so we only train prompts
        for p in self.clip.parameters():
            p.requires_grad = False
        
        self.text_embedding_layer = self.clip.text_model.embeddings
        self.init_context_learner()
        self.use_fresh_lr_scheduler = config["use-fresh-lr-scheduler"]

        # optimizer over prompt params only
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()

        # For flop analysis
        self._modules ={
            "clip": self.clip,
            "context_learner": self.context_learner
        }

    def init_context_learner(self):
        self.context_learner = TextContextLearner(
            self.tokenizer,
            embed_dim_text=self.clip.config.text_config.hidden_size,
            embed_dim_vision=self.clip.config.vision_config.hidden_size,
            class_names=self.class_names,
            config=self.config,
            device=self.device
            ).to(self.device)
        return self.context_learner

    def get_optimizer(self):
        params = [p for p in self.context_learner.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.config["lr"], weight_decay=self.config["weight-decay"]) # type: ignore

    def get_lr_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config["epochs"], eta_min=1e-6) # type: ignore

    def forward(
            self, 
            videos: torch.Tensor, 
            video_masks: Optional[torch.BoolTensor] = None, 
            text_features: Optional[torch.Tensor] = None,
            class_names: Optional[list[str]] = None, 
            softmax: bool = False,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the logits for a batch of videos (B, V, T, C, H, W) and the given classes (or those used in training if None).
        Also return the text and video features.
        Returns:
            logits: (batch_size, num_classes)
            text_features: (num_classes, D)
            video_features: (batch_size, D)
        """
        # Should already be on device so call is for safety
        videos = videos.to(self.device)
        video_masks = video_masks.to(self.device) if video_masks is not None else None

        video_features = self.encode_video_batch(pixel_values=videos, mask=video_masks)
        text_features = self.encode_texts(class_names=class_names) if text_features is None else text_features  # here to ensure fresh backprop graph during training
        # Normalise features
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = self._get_logits(text_features, video_features, softmax=softmax)

        return logits, text_features, video_features
    
    def encode_video_batch(self, pixel_values: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None, normalize: bool = True, temporally_pool: bool = True, return_attentions: bool = False) -> torch.FloatTensor:
        """
        Encode every frame of each video by combining the batch and temporal dimensions. 
        Apply temporal pooling and normalise by default.
        Parameters:
        pixel_values: (B, V, T, C, H, W)
        mask: Optional boolean mask (B, V, T) indicating valid frames
        normalize: whether to L2-normalize the output features
        temporally_pool: whether to apply temporal pooling
        Returns:
        vid_feats: (B, V, D) if temporally_pool else (B, V, T, D)
        """
        B, V, T, C, H, W = pixel_values.shape

        reshaped_pixels = pixel_values.contiguous().view(B*V*T, C, H, W)
        
        if return_attentions:
            # ---- attention path ----
            outputs = self.clip.vision_model(
                pixel_values=reshaped_pixels.to(self.device, self.clip.dtype),
                output_attentions=True,
                output_hidden_states=False,
            )

            # CLS token
            hidden = outputs.last_hidden_state[:, 0]
            hidden = self.clip.vision_model.post_layernorm(hidden)
            vid_feats = self.clip.visual_projection(hidden)

            # (L, B*V*T, H, N, N)
            attentions = torch.stack(outputs.attentions, dim=0)
        else:
            vid_feats = self.clip.get_image_features(pixel_values=reshaped_pixels.to(self.device, self.clip.dtype))

        if normalize:
            vid_feats = vid_feats / vid_feats.norm(dim=-1, keepdim=True)

        if temporally_pool:
            vid_feats = self._temporally_pool_with_mask(vid_feats, B, V, T, mask)
        else:
            vid_feats = vid_feats.view(B, V, T, -1)

        if return_attentions:   
            return vid_feats, attentions
        return vid_feats

    def encode_texts(self, class_names: Optional[list[str]] = None, normalise: bool = True) -> torch.FloatTensor:
        """
        Encode text prompts as vectors for the given classes using the context vectors.
        """
        # builds text features for all classes via the prompt learner
        inputs_embeds, attention_mask = self.context_learner.get_text_context_embeds(self.text_embedding_layer, class_names=class_names)
        # pass each prompt through text encoder; CLIP TextModel accepts inputs_embeds and attention_mask
        outputs = self.clip.text_model.encoder(inputs_embeds,) # attention mask for padding not used attention_mask=attention_mask.to(self.device)
        last_hidden_state = outputs.last_hidden_state

        pooled_output = self._get_pooled_text_output(last_hidden_state, attention_mask)
        # if CLIPModel has a text projection layer, apply it
        if hasattr(self.clip, 'text_projection'):
            text_features = self.clip.text_projection(pooled_output)
        print(f"text features stats: {text_features.mean().item():.3e} ± {text_features.std().item():.3e}, min: {text_features.min().item():.3e}, max: {text_features.max().item():.3e}") if self.debug else None
        if normalise:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _get_logits(self, text_features, video_features, softmax=False, view_pooling=True):
        """
        video_features: (B, V, D)
        text_features:  (n_classes, D) or (B, V, n_classes, D)
        returns: (B, n_classes) if view_pooling else (B, V, n_classes)
        """

        B, V, D = video_features.shape
        with torch.no_grad():
            self.context_learner.logit_scale.clamp_(max=np.log(100))
        logit_scale = self.context_learner.logit_scale.exp()

        if text_features.ndim == 2:
            # (n_classes, D) → (B, V, n_classes, D)
            text_features = text_features.unsqueeze(0).unsqueeze(0)  # (1,1,n_classes,D)
            text_features = text_features.expand(B, V, -1, -1)       # (B,V,n_classes,D)

        # always (B, V, n_classes, D)
        logits = logit_scale * torch.einsum('bvd,bvcd->bvc', video_features, text_features)

        if view_pooling:
            logits = logits.mean(dim=1)  # (B, n_classes)

        if softmax:
            logits = logits.softmax(dim=-1)

        return logits

    def _get_pooled_text_output(self, last_hidden_state, attention_mask):
        """
        Supports:
            (n_classes, n_tokens, D)
            (B, V, n_classes, n_tokens, D)

        Returns pooled over tokens → shape (same leading dims..., n_classes, D)
        """
        squeeze_dims = False
        # Normalize to always 5D internally
        if last_hidden_state.ndim == 3:
            # (n_classes, n_tokens, D) → (1, 1, n_classes, n_tokens, D)
            last_hidden_state = last_hidden_state.unsqueeze(0).unsqueeze(0)
            attention_mask    = attention_mask.unsqueeze(0).unsqueeze(0)
            squeeze_dims = True
            
        B, V, n_classes, n_tokens, D = last_hidden_state.shape

        x = last_hidden_state.view(B * V * n_classes, n_tokens, D)
        m = attention_mask.view(B * V * n_classes, n_tokens)

        seq_lens = m.sum(dim=1, dtype=torch.long) - 1  # last non-pad token index
        pooled   = x[torch.arange(x.size(0), device=x.device), seq_lens]  # (B*V*n_classes, D)
        # reshape back, then drop added batch/view dims only if they were originally absent
        return pooled.view(B, V, n_classes, D).squeeze(0).squeeze(0) if squeeze_dims else pooled.view(B, V, n_classes, D)

    
    def _temporally_pool_with_mask(self, vid_feats: torch.FloatTensor, B: int, V: int, T: int, mask: Optional[torch.BoolTensor] = None, temporal_dim=2) -> torch.FloatTensor:
        """
        Pool the temporal dimension of the features according to the configured method with the mask.
        Takes in features of shape (B*V*T, D) and returns (B, V, D).
        """
        if mask is not None:
            mask_reshaped = mask.view(B * V * T, -1)
            vid_feats = vid_feats * mask_reshaped

        vid_feats = vid_feats.contiguous().view(B, V, T, -1)

        vid_feats = self.context_learner.temporal_pooling(vid_feats, dim=temporal_dim)
        
        return vid_feats

    def train_epoch(self, dataloader: DataLoader, wandb_step: int, epoch: Optional[int] = None, output_folder: Optional[str] = None):
        self.context_learner.train()
        total_loss, correct, total = 0.0, 0, 0

        hand_crafted_features = self.context_learner.get_hand_crafted_features(self.clip).detach()
        self.context_learner.post_backward_computations()

        counter = 0
        self.optimizer.zero_grad()                      
        pbar = tqdm(enumerate(dataloader),
                    f"Train Dataloader" + (f" {epoch}" if epoch is not None else ""),
                    total=len(dataloader), mininterval=self.min_interval_tqdm)
        
        for i, batch in pbar:
            if batch is None:
                print(f"[WARN] Skipping None batch at step {i}")
                continue

            videos, masks, labels = batch
            videos = videos.to(self.device)
            masks = masks.to(self.device) if masks is not None else None
            labels = labels.to(self.device)

            # --- Debug: check inputs ---
            if self.debug:
                if torch.isnan(videos).any() or torch.isinf(videos).any():
                    print(f"[ERROR] NaN/Inf in videos at step {i}")
                if torch.isnan(labels).any():
                    print(f"[ERROR] NaN in labels at step {i}")

            with torch.amp.autocast(enabled=self.config["fp16"], device_type=self.device):
                logits, text_features, video_features = self.forward(
                    videos=videos,
                    video_masks=masks,
                    text_features=hand_crafted_features if self.use_handcrafted_features else None,
                    class_names=None,
                    softmax=False,
                )

                if self.debug:
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"[ERROR] NaN/Inf in logits at step {i}")

                loss = self.loss_fn(logits, labels)
                if torch.isnan(loss):
                    print(f"[ERROR] NaN loss before regularization at step {i}")

                if self.regularisation_strength != 0:
                    reg_loss = regularisation_loss(hand_crafted_features, text_features).to(self.device)
                    if torch.isnan(reg_loss):
                        print(f"[ERROR] NaN in reg_loss at step {i}")
                    loss = loss + reg_loss * self.regularisation_strength

                if self.cosine_regularisation != 0:
                    cos_loss = cosine_similarity_loss(hand_crafted_features, text_features).to(self.device)
                    if torch.isnan(cos_loss):
                        print(f"[ERROR] NaN in cos_loss at step {i}")
                    loss = loss + cos_loss * self.cosine_regularisation

                loss = loss / self.batches_per_backprop

            # --- Debug: Gradients ---
            self.scaler.scale(loss).backward()
            self.context_learner.clear_pre_computations()
            self.context_learner.post_backward_computations()

            if i % 50 == 0 and self.debug:  # every few steps
                total_norm = 0.0
                for p in self.context_learner.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"[DEBUG] Step {i}: loss={loss.item():.4f}, grad_norm={total_norm:.2f}")
            torch.nn.utils.clip_grad_norm_(self.context_learner.parameters(), self.grad_norm_max)

            counter += 1
            if counter % self.batches_per_backprop == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            loss_true = loss * self.batches_per_backprop
            total_loss += loss_true * videos.size(0)
            correct_batch, total_batch = self.acc_fn(logits, labels)
            correct += correct_batch
            total += total_batch

            if wandb_step % self.config["log-every-n-steps"] == 0:
                total_loss = total_loss.item()
                acc = correct.item() / total
                pbar.set_postfix(loss=total_loss / self.batches_per_backprop / videos.size(0) / (i + 1),
                                acc=acc,
                                video_T=videos.shape[2])

                if torch.isnan(torch.tensor(total_loss)):
                    print(f"[ERROR] Accumulated loss is NaN at step {i}")

            if self.wandb_logger and wandb_step % self.config["log-every-n-steps"] == 0:
                acc = correct.item() / total
                self.wandb_logger.log({
                    "train_loss_batch": loss_true.item(),
                    "train_acc_batch": acc,
                    "learning_rate": self.lr_scheduler.get_last_lr()[0],
                    "step": wandb_step,
                })
            wandb_step += 1

            if self.config["save-every-n-steps"] is not None and wandb_step % self.config["save-every-n-steps"] == 0 and output_folder is not None:
                acc = correct.item() / total
                self.save_context(os.path.join(output_folder, f"{epoch}_{wandb_step}.pt"), meta_data={"epoch": epoch, "train_acc": acc})


        # flush any remaining gradients (works with scaler enabled/disabled)
        if counter % self.batches_per_backprop != 0:
            # use scaler consistent API
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        acc = correct.item() / total
        total_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return total_loss / total, acc

    @torch.no_grad()
    def eval(self, dataloader: DataLoader, epoch: Optional[int] = None, class_names: Optional[list[str]] = None, use_hand_crafted_text_features: bool = False) -> tuple[float, float, torch.Tensor, torch.Tensor]:
        """
        Evaluate on the given dataloader. If it contains a different set of classes to those used in training,
        these must be passed as to compute the text features for the new classes.
        """
        if class_names is None:
            class_names = self.class_names
        self.context_learner.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_logits, all_labels = [], []

        pbar = tqdm(enumerate(dataloader), f"Eval Dataloader" + (f" {epoch}" if epoch is not None else ""), total=len(dataloader), mininterval=self.min_interval_tqdm)

        if use_hand_crafted_text_features:
            text_features = self.context_learner.get_hand_crafted_features(self.clip).detach()
        else:
            text_features = self.encode_texts(class_names=class_names)

        for i, batch in pbar:
            if batch is None:
                print(f"[WARN] Skipping None batch during eval at step {i}")
                continue
            videos, masks, labels = batch
            videos = videos.to(self.device)
            masks = masks.to(self.device) if masks is not None else None
            labels = labels.to(self.device)

            with torch.amp.autocast(enabled=self.config["fp16"], device_type=self.device):
                logits, _, _ = self.forward(
                    videos=videos,
                    text_features=text_features,
                    video_masks=masks,
                    class_names=class_names,
                    softmax=False
                )

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * videos.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += videos.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            pbar.set_postfix(loss=loss.item(), acc=correct/total)

        avg_loss = total_loss / total
        acc = correct / total

        # Raw confusion matrix
        logits_cat = torch.cat(all_logits).detach()
        labels_cat = torch.cat(all_labels).detach()

        if self.wandb_logger is not None:
            calculate_class_metrics(
                logits_cat=logits_cat,
                labels_cat=labels_cat,
                class_names=class_names,
                logger=self.wandb_logger,
                loss=avg_loss
            )

        return avg_loss, acc, logits_cat, labels_cat

    def train(self, train_loader: DataLoader, val_loader: DataLoader, output_folder: str, epochs: Optional[int] = None, train_loader1: Optional[DataLoader] = None):
        best_val = 0.0
        pbar = tqdm(range(1, self.config["epochs"] + 1) if epochs is None else epochs, "Training")
        self.save_config(os.path.join(output_folder, "config.json"))
        best_path = os.path.join(output_folder, "best.pt")
        # self.save_context(os.path.join(output_folder, f"untrained.pt"))
        if self.config["learn-just-pooling"]:
            print("Training only temporal pooling layer as per config.")
            for param in self.context_learner.parameters():
                param.requires_grad = False
            for param in self.context_learner.temporal_pooling.parameters():
                param.requires_grad = True
            self.optimizer = self.get_optimizer()
            self.lr_scheduler = self.get_lr_scheduler()

        wandb_step = 0
        early_stop_counter = 0
        use_train_loader_1 = False if train_loader1 is None else True
        for ep in pbar:
            if ep <= self.epochs_to_skip:
                print(f"Skipping epoch {ep} as per config")
                self.lr_scheduler.step()
                continue
            
            if use_train_loader_1:
                print(f"Using train_loader1 from video {self.config['start-train-step'] * self.config['batch-size']} to {len(train_loader1.dataset)} for first epoch")
                train_loss, train_acc = self.train_epoch(train_loader1, wandb_step=wandb_step, epoch=ep, output_folder=output_folder)
                use_train_loader_1 = False
            else:       
                train_loss, train_acc = self.train_epoch(train_loader, wandb_step=wandb_step, epoch=ep, output_folder=output_folder)

            with torch.no_grad():
                val_loss, val_acc, _, _ = self.eval(val_loader, epoch=ep, class_names=self.eval_class_names)
            print(f"Epoch {ep}: train loss={train_loss:.4f} acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f}")

            self.config["training-history"]["train-acc"].append(train_acc)
            self.config["training-history"]["train-loss"].append(train_loss)
            self.config["training-history"]["val-acc"].append(val_acc)
            self.config["training-history"]["val-loss"].append(val_loss)

            self.save_config(os.path.join(output_folder, "config.json"))
            self.save_context(os.path.join(output_folder, f"{ep}.pt"), meta_data={"epoch": ep, "val_acc": val_acc, "train_acc": train_acc})

            pbar.set_postfix_str(f"train loss={train_loss:.4f} acc={train_acc:.4f} | val loss={val_loss:.4f} acc={val_acc:.4f}")
            
            if self.wandb_logger is not None:
                self.wandb_logger.log({
                    "epoch": ep,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "step": wandb_step,
                })
            
            # save best and early stopping
            if val_acc > best_val:
                best_val = val_acc
                self.save_context(best_path)
                early_stop_counter = 0
                print(f"New best val acc: {best_val:.4f}, saved to {best_path}.")
            else:
                early_stop_counter += 1
                print(f"No improvement in val acc for {early_stop_counter} epochs.")
                if early_stop_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered after {self.early_stop_patience} epochs without improvement.")
                    break

            self.lr_scheduler.step()
        
        print(f"Best val acc: {best_val:.4f}.")

    def save_context(self, path: str, meta_data: dict = {}, epoch: Optional[int] = None):
        checkpoint = {
            'context_state_dict': self.context_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'epoch': epoch,
            'training_history': self.config.get("training-history", {}),
            'config': self.config
        }
        torch.save(checkpoint, path)
        
        if self.wandb_logger is not None:
            dirname = os.path.basename(os.path.dirname(path))
            basename = os.path.basename(path)
            art = wandb.Artifact(name=f"{dirname}-{basename}", type="checkpoint", metadata=meta_data)
            art.add_file(path)
            wandb.log_artifact(art)

    def load_context(self, path: str):
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        try:
            self.context_learner.load_state_dict(checkpoint['context_state_dict'])
            print(f"Loaded context from {path}")
        except Exception as e:
            print(f"Error loading checkpoint {path}: {e}. Will try old method for context_learner.")
            # Training fails to continue with the old method so we just load the context_learner weights
            try:
                self.context_learner.load_context(path)
            except Exception as e2:
                print(f"Failed to load context from {path}: {e2}")
                return
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded optimiser from {path} (epoch {checkpoint.get('epoch', 'N/A')})")
        except Exception as e:
            print(f"Error loading optimizer from {path}: {e}. Continuing without loading optimizer state")
        try:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict']) if not self.use_fresh_lr_scheduler else None
            print(f"Loaded LR scheduler from {path}")
        except Exception as e:
            print(f"Error loading LR scheduler from {path}: {e}, continuing without loading it.")
        
        try:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Loaded Grad Scaler from {path}")
        except Exception as e:
            print(f"Error loading Grad Scaler from {path}: {e}, continuing without loading it.")
        
        try:
            self.config["training-history"] = checkpoint.get("training_history", {})
        except Exception as e:
            print(f"Error loading training history from {path}: {e}, continuing without loading it.")
        
        return checkpoint.get("epoch", None)

    def save_config(self, path):
        save_config(path, self.config)
        if self.wandb_logger is not None:
            art = wandb.Artifact(name=os.path.basename(path), type="config")
            art.add_file(path)
            wandb.log_artifact(art)
    
    def eval_mode(self):
        self.context_learner.eval()
        self.clip.eval()

    def quantize_model(self, dtype=torch.float16):
        self.context_learner = torch.quantization.quantize_dynamic(
            self.context_learner, {nn.Linear}, dtype=dtype
        )

    def warmup(self):
        B, V, T, C, H, W = 1, 1, self.config["num-frames"], 3, self.processor.image_processor.crop_size["height"], self.processor.image_processor.crop_size["width"]
        dummy_video = torch.randn(B, V, T, C, H, W).to(self.device)
        dummy_mask = torch.ones(B, V, T, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            self.forward(videos=dummy_video, video_masks=dummy_mask)
    
    def to(self, device: str):
        self.device = device
        self.clip.to(device)
        self.context_learner.to(device)