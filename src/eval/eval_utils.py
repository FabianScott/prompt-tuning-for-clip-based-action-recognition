"""
Shared utilities for evaluation scripts to reduce code duplication.
"""
import os
import json
import datetime
import wandb
import numpy as np
from typing import Optional, Tuple
from transformers import CLIPProcessor

from ..configs.paths import EVAL_PATH, CLIP_MODEL_CACHE_DIR
from ..configs.read_token import read_token
from ..configs.read_config import get_checkpoint_path, read_config_from_file, read_dataset_config
from ..data.hugginface_utils import huggingface_login
from ..data.dataloading import build_dataloader
from ..data.dataset_builders import build_dataset, build_balanced_subset


def setup_evaluation_environment():
    """Setup common environment variables and authentication."""
    huggingface_login()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def generate_test_path(eval_config: dict, prefix: str = "") -> Tuple[str, str]:
    """
    Generate standardized test name and save path with timestamp.
    
    Args:
        eval_config: Evaluation configuration dict
        prefix: Optional prefix for the test name (e.g., 'calibration', 'mixture')
    
    Returns:
        Tuple of (test_name, metrics_save_path)
    """
    run_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
    
    components = []
    if prefix:
        components.append(prefix)
    
    # Add standard path components
    if eval_config.get('use-augmentation') is not None:
        components.append("augmentation-" + str(eval_config['use-augmentation']))
    if eval_config.get('model_name'):
        components.append(eval_config['model_name'])
    if eval_config.get('dataset_train'):
        components.append(eval_config['dataset_train'])
    if eval_config.get('dataset-test'):
        components.append(eval_config['dataset-test'])
    elif eval_config.get('dataset_test'):
        components.append(eval_config['dataset_test'])
    
    components.append(eval_config['run_id'])
    components.append(run_time_str)
    
    test_name = "/".join(components)
    metrics_save_path = os.path.join(EVAL_PATH, test_name)
    
    return test_name, metrics_save_path


def load_config_and_checkpoint(eval_config: dict, model_key: Optional[str] = None) -> Tuple[dict, str]:
    """
    Load method configuration and checkpoint path.
    
    Args:
        eval_config: Evaluation configuration
        model_key: Optional model key in format "method/dataset/runid"
    
    Returns:
        Tuple of (method_config, checkpoint_path)
    """
    if model_key:
        name, dataset, runid = model_key.split("/")
        checkpoint_path = get_checkpoint_path(
            method_name=name,
            dataset_name=dataset,
            run_id=runid,
        )
    else:
        checkpoint_path = get_checkpoint_path(
            method_name=eval_config["model_name"],
            dataset_name=eval_config["dataset_train"],
            run_id=eval_config["run_id"],
        )
    
    dirname = os.path.dirname(checkpoint_path)
    config_path = os.path.join(dirname, "config.json")
    method_config = read_config_from_file(config_path)
    
    return method_config, checkpoint_path


def prepare_method_config(method_config: dict, eval_config: dict, base_config: Optional[dict] = None) -> dict:
    """
    Update method config with eval config and base config.
    
    Args:
        method_config: Method configuration to update
        eval_config: Evaluation configuration
        base_config: Optional base configuration
    
    Returns:
        Updated method_config
    """
    # Get dataset test name
    dataset_test = eval_config.get('dataset-test') or eval_config.get('dataset_test')
    
    # Update dataset config
    base_dataset_config = read_dataset_config(dataset_test)
    method_config["dataset-config"].update(base_dataset_config)
    method_config["dataset"] = dataset_test
    
    # Apply eval config overrides
    for key in eval_config:
        if key in method_config["dataset-config"]:
            method_config["dataset-config"][key] = eval_config[key]
        elif base_config and eval_config.get("use-augmentation") and key in base_config.get("augmentation-dict", {}):
            base_config["augmentation-dict"][key] = eval_config[key]
        elif key in method_config:
            method_config[key] = eval_config[key]
        elif base_config and key not in method_config and key in base_config:
            method_config[key] = eval_config[key]
    
    if base_config:
        base_config.update(method_config)
        method_config = base_config
    
    return method_config


def build_evaluation_dataset(eval_config: dict, method_config: dict):
    """
    Build dataset for evaluation.
    
    Args:
        eval_config: Evaluation configuration
        method_config: Method configuration
    
    Returns:
        Tuple of (dataset, class_names)
    """
    dataset_name = eval_config.get('dataset-test') or eval_config.get('dataset_test')
    split = eval_config.get('split', 'test')
    
    tmp_processor = CLIPProcessor.from_pretrained(
        method_config["clip-model"], 
        cache_dir=CLIP_MODEL_CACHE_DIR
    )
    
    dataset = build_dataset(
        name=dataset_name,
        split=split,
        processor=tmp_processor,
        config=method_config
    )
    
    class_names = dataset.class_names
    
    # Handle balanced subset if K-test specified
    k_test = eval_config.get('K-test')
    if k_test is not None:
        dataset, class_names = build_balanced_subset(
            dataset=dataset,
            examples_per_class=k_test,
        )
    
    return dataset, class_names


def build_evaluation_dataloader(dataset, meta_config: dict):
    """
    Build dataloader for evaluation.
    
    Args:
        dataset: Dataset to create loader for
        meta_config: Meta configuration with batch-size and num-workers
    
    Returns:
        DataLoader
    """
    return build_dataloader(
        dataset=dataset,
        batch_size=meta_config.get("batch-size", 1),
        num_workers=meta_config.get("num-workers", 0),
    )


def init_wandb_logger(meta_config: dict, method_config: dict, test_name: str, tags: list):
    """
    Initialize wandb logger if enabled.
    
    Args:
        meta_config: Meta configuration
        method_config: Method configuration
        test_name: Name for the run
        tags: Tags for the run
    
    Returns:
        wandb run or None
    """
    if not meta_config.get("use-wandb", False):
        return None
    
    wandb.login(key=read_token('tokens/wandb.txt', 'Weights & Biases'))
    return wandb.init(
        project="Thesis - Video Classification",
        config=method_config,
        name=test_name,
        tags=tags
    )


def save_metrics_json(metrics: dict, save_path: str):
    """
    Save metrics to JSON file, converting numpy types to Python types.
    
    Args:
        metrics: Dictionary of metrics to save
        save_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    metrics_converted = convert_numpy(metrics)
    
    with open(save_path, "w") as f:
        json.dump(metrics_converted, f, indent=4)


def load_or_compute_metrics(save_path: str, compute_fn, force_rerun: bool = False):
    """
    Load metrics from disk if they exist, otherwise compute them.
    
    Args:
        save_path: Path to metrics JSON file
        compute_fn: Function to call if metrics need to be computed
        force_rerun: If True, always recompute even if metrics exist
    
    Returns:
        Metrics dictionary
    """
    if os.path.exists(save_path) and not force_rerun:
        print(f"Loading existing metrics from {save_path}")
        with open(save_path, "r") as f:
            return json.load(f)
    else:
        print(f"Computing metrics (will save to {save_path})")
        return compute_fn()
