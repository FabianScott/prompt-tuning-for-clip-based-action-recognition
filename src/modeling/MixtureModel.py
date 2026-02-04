import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from .CoOp import CoOpModel
from .utils import calculate_class_metrics
from ..plots import plot_metrics_grid
from ..configs.parse_run_args import load_checkpoint_given_path, load_model_from_keys
from ..configs.read_config import read_config_from_file
from ..eval.eval_utils import (
    setup_evaluation_environment,
    generate_test_path,
    load_config_and_checkpoint,
    prepare_method_config,
    build_evaluation_dataset,
    build_evaluation_dataloader,
    init_wandb_logger,
    save_metrics_json
)


def run_mixture_evaluation(
    eval_config,
    meta_config,
    key_paths: list[str],
    weights: Optional[list[float]] = None,
    wandb_logger = None,
):
    test_name, metrics_save_path = generate_test_path(eval_config, prefix="mixture")
    
    setup_evaluation_environment()
    
    # Load config from first model
    method_config, _ = load_config_and_checkpoint(eval_config, model_key=key_paths[0])
    method_config = prepare_method_config(method_config, eval_config)
    method_config["mixture-sources"] = key_paths
    method_config["mixture-weights"] = weights
    
    # Build dataset and dataloader
    dataset, class_names = build_evaluation_dataset(eval_config, method_config)
    dataloader = build_evaluation_dataloader(dataset, meta_config)
    
    # Initialize wandb
    wandb_logger = init_wandb_logger(meta_config, method_config, "Mixture-" + test_name, ["mixture-model"])
    
    # Load models
    models = [
        load_model_from_keys(
            keys=key,
            wandb_logger=None,
            class_names=class_names,
        )
        for key in key_paths
    ]

    mixture = MixtureModel(models=models, weights=weights, wandb_logger=wandb_logger)

    loss, acc, logits_cat, labels_cat = mixture.eval(
        dataloader=dataloader,
        class_names=class_names,
        use_hand_crafted_text_features=eval_config.get("use-handcrafted-prompts", False),
    )

    scores_per_class = calculate_class_metrics(
        logits_cat=logits_cat,
        labels_cat=labels_cat,
        class_names=class_names,
        logger=None,
        loss=loss,
    )

    save_metrics_json(scores_per_class, os.path.join(metrics_save_path, "metrics_per_class.json"))

    plot_metrics_grid(
        scores_per_class,
        savename=test_name + f"{eval_config['run_id']}.png",
        bins=20,
    )


def build_mixture_from_sources(sources, weights=None, device="cuda"):
    """
    sources: list of checkpoint paths OR config.json paths
    weights: list of floats
    """
    if weights is None:
        weights = [1.0 / len(sources)] * len(sources)
    assert len(weights) == len(sources), "Weights length must match sources length"
    models = []

    for src in sources:
        if src.endswith(".json"):  # config path
            config = read_config_from_file(src)
            ckpt = config.get("continue-from", None)
            if ckpt is None:
                raise ValueError(f"No checkpoint in config {src}")
            model = load_checkpoint_given_path(
                method_name=config["model_name"],
                checkpoint_path=ckpt,
                method_config=config,
                device=device,
                class_names=None,
                wandb_logger=None,
            )
        else:  # checkpoint path
            dirname = os.path.dirname(src)
            config = read_config_from_file(os.path.join(dirname, "config.json"))
            model = load_checkpoint_given_path(
                method_name=config["model_name"],
                checkpoint_path=src,
                method_config=config,
                device=device,
                class_names=None,
                wandb_logger=None,
            )

        models.append(model)

    return MixtureModel(models=models, weights=weights), config


class MixtureModel(CoOpModel):
    def __init__(self, models: list[CoOpModel], weights: Optional[list[float]] = None, wandb_logger=None):
        # inherit config / classes from first model
        base = models[0]
        super().__init__(
            config=base.config,
            class_names=base.class_names,
            device=base.device,
            eval_class_names=base.eval_class_names,
            wandb_logger=wandb_logger,
        )

        if weights is None: 
            w = torch.ones(len(models), dtype=torch.float32) / len(models)
        else:
            w = torch.tensor(weights, dtype=torch.float32)
        self.weights = (w / w.sum()).to(base.device)

        # ensure eval-only
        self.models = models
        for m in self.models:
            m.eval_mode()

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
            text_features = [
                m.encode_texts(class_names=class_names) for m in self.models
            ]
        for i, batch in pbar:
            if batch is None:
                print(f"[WARN] Skipping None batch during eval at step {i}")
                continue
            videos, masks, labels = batch
            videos = videos.to(self.device)
            masks = masks.to(self.device) if masks is not None else None
            labels = labels.to(self.device)

            with torch.amp.autocast(enabled=self.config["fp16"], device_type=self.device):
                for model_idx, model in enumerate(self.models):
                    model_logits, _, _ = model.forward(
                        videos=videos,
                        text_features=text_features[model_idx],
                        video_masks=masks,
                        class_names=class_names,
                        softmax=True
                    )
                    if model_idx == 0:
                        logits = self.weights[model_idx] * model_logits
                    else:
                        logits += self.weights[model_idx] * model_logits

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
