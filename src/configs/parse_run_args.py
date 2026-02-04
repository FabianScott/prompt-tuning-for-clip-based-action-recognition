# --------------------------- Dataset & DataLoader ---------------------------
# --------------------------- Trainer ---------------------------
# --------------------------- CLI & main -------------------------------------

import os
import torch
import argparse

from . import return_combined_config
from .datasets_and_methods import dataset_names, method_names
from .paths import hmdb51_root, ucf_101_root, nwpu_root, kinetics400_root, DATA_PROCESSED_PATH
from ..modeling.CoOp import CoOpModel
from .read_config import read_config_from_file, get_checkpoint_path

def get_dataset_root(dataset: str):
    dataset = dataset.lower()
    if dataset == "ucf101":
        return ucf_101_root
    elif dataset == "hmdb51":
        return hmdb51_root
    elif dataset == "nwpu":
        return nwpu_root
    elif dataset == "kinetics400":
        return kinetics400_root
    else:
        raise ValueError(f"Unknown dataset: {dataset}, should be one of {dataset_names}")

def parse_run_args(method_default: str, dataset_default: str, default_config: dict = None, override_args: dict = None):
    p = argparse.ArgumentParser()
    p.add_argument('--prompt-optimisation-method', type=str, default=method_default, help=f'Prompt optimisation method to use for training, one of {method_names}')
    p.add_argument('--dataset', type=str, default=dataset_default, help=f'Dataset to use, one of {dataset_names}')
    args, _ = p.parse_known_args()
    method_config = return_combined_config(
        args.prompt_optimisation_method, 
        args.dataset, 
        use_augmentation=override_args.get("use-augmentation", False) if override_args is not None else False
        ) if default_config is None else default_config
    if override_args is not None:
        if "dataset-config" in override_args:
            method_config["dataset-config"].update(override_args["dataset-config"])
            del override_args["dataset-config"]
        method_config.update(override_args)
    data_root_default = get_dataset_root(args.dataset)

    p.add_argument('--data-dir', type=str, default=data_root_default, help='Path to video data in Pytorch Dataset format', )
    p.add_argument('--model', type=str, default=method_config["clip-model"], help='Name of the CLIP model variant to use, e.g. openai/clip-vit-base-patch32')
    p.add_argument('--K-train', type=int, default=method_config["dataset-config"]["K-train"], help="Number of training examples per class for K-shot learning.")
    p.add_argument('--K-val', type=int, default=method_config["dataset-config"]["K-val"], help="Number of validation examples per class for K-shot learning.")
    p.add_argument('--K-test', type=int, default=method_config["dataset-config"]["K-test"], help="Number of test examples per class for K-shot learning.")
    p.add_argument('--val-examples-per-class', type=int, default=method_config["dataset-config"]["K-val"])
    p.add_argument('--epochs', type=int, default=method_config["epochs"])
    p.add_argument('--batch-size', type=int, default=method_config["dataset-config"]["batch-size"])
    p.add_argument('--batches-per-backprop', type=int, default=method_config["dataset-config"]["batches-per-backprop"])
    p.add_argument('--lr', type=float, default=method_config["lr"])
    p.add_argument('--weight-decay', type=float, default=method_config["weight-decay"])
    p.add_argument('--regularisation-strength', type=float, default=method_config["regularisation-strength"])
    p.add_argument('--regularisation-text', type=str, default=method_config["regularisation-text"])
    p.add_argument('--cosine-regularisation-strength', type=float, default=method_config["cosine-regularisation-strength"])
    p.add_argument('--ctx-len', type=int, default=method_config["ctx-len"])
    p.add_argument('--std-init', type=float, default=method_config["std-init"])
    p.add_argument('--class-wise', action='store_true', help="Use class-wise prompts", default=method_config["class-wise"])
    p.add_argument('--no-class-wise', dest='class_wise', action='store_false')
    p.add_argument('--out', type=str, default=DATA_PROCESSED_PATH + f'prompts/{args.prompt_optimisation_method}/{args.dataset}')
    p.add_argument('--num-workers', type=int, default=method_config["dataset-config"]["num-workers"])
    p.add_argument('--split-file', type=str, default=method_config["dataset-config"]["split-file"])
    p.add_argument('--split-function', type=str, default=method_config["dataset-config"]["split-function"])
    p.add_argument('--random-seed', type=int, default=method_config["dataset-config"]["random-seed"])
    p.add_argument('--has-validation', action='store_true', help="Use validation set", default=method_config["dataset-config"]["has-validation"])
    p.add_argument('--no-validation', dest='has_validation', action='store_false')
    p.add_argument('--use-cache', action='store_true', help="Use cached dataset", default=method_config["dataset-config"]["use-cache"])
    p.add_argument('--no-cache', dest='use_cache', action='store_false')
    p.add_argument('--save-to-cache', action='store_true', help="Save processed data to cache", default=method_config["dataset-config"]["save-to-cache"])
    p.add_argument('--no-save-cache', dest='save_to_cache', action='store_false')
    p.add_argument('--resized-cache-protocol', type=str, default=method_config["dataset-config"]["resized-cache-protocol"])
    p.add_argument('--class-protocol', type=str, default=method_config["dataset-config"]["class-protocol"])
    p.add_argument('--num-classes', type=int, default=method_config["dataset-config"]["num-classes"])
    p.add_argument('--use-excess-for', type=str, default=method_config["dataset-config"]["use-excess-for"])
    p.add_argument('--remove-class-if-insufficient', action='store_true', help="Remove classes with insufficient samples", default=method_config["dataset-config"]["remove-class-if-insufficient"])
    p.add_argument('--no-remove-insufficient', dest='remove_class_if_insufficient', action='store_false')
    p.add_argument('--skip-first-K-train', type=int, default=method_config["dataset-config"]["skip-first-K-train"])
    p.add_argument('--skip-first-K-val', type=int, default=method_config["dataset-config"]["skip-first-K-val"])
    p.add_argument('--val-proportion', type=float, default=method_config["dataset-config"]["val-proportion"])
    p.add_argument('--use-clips', action='store_true', help="Use video clips instead of full videos", default=method_config["dataset-config"]["use-clips"])
    p.add_argument('--no-clips', dest='use_clips', action='store_false')
    p.add_argument('--remove-black-strips', action='store_true', help="Remove black strips from videos", default=method_config["dataset-config"]["remove-black-strips"])
    p.add_argument('--no-remove-black-strips', dest='remove_black_strips', action='store_false')
    p.add_argument('--re-process-data', type=bool, default=False, help="Whether to re-download and process the dataset even if cached version exists.")
    p.add_argument('--use-wandb', action='store_true', help="Enable W&B logging", default=True)
    p.add_argument('--no-wandb', dest='use_wandb', action='store_false')
    p.add_argument('--continue-from', type=str, default=method_config["continue-from"], help="Path to checkpoint or name in checkpoint.json to continue training from.")
    p.add_argument('--start-train-step', type=int, default=method_config["start-train-step"], help="Step to start training from")
    p.add_argument('--save-every-n-steps', type=float, default=method_config["save-every-n-steps"], help="Save checkpoint every n steps")
    p.add_argument('--checkpoint-last-n', type=int, default=method_config["checkpoint-last-n"], help="Keep only the last n checkpoints")
    p.add_argument('--early-stop-patience', type=int, default=method_config["early-stop-patience"], help="Number of epochs to wait before early stopping")
    p.add_argument('--loss-function', type=str, default=method_config["loss-function"], help="Loss function to use")
    p.add_argument('--temporal-pooling', type=str, default=method_config["temporal-pooling"], help="Temporal pooling method to use, one of 'mean', 'max', 'attention'")    
    p.add_argument('--num-head-attention-pooling', type=int, default=method_config["num-heads-attention-pooling"], help="Number of attention heads to use if temporal pooling is 'attention'")
    p.add_argument('--num-frames-temporal', type=int, default=method_config["num-frames-temporal"], help="Number of frames to sample temporally from each video")
    p.add_argument('--space-between-frames', type=int, default=method_config["space-between-frames"], help="Number of frames to skip between sampled frames, e.g. 2 means sample every 3rd frame")
    p.add_argument('--num-temporal-views', type=int, default=method_config["num-temporal-views"], help="Number of temporal clips to sample from each video during training")
    p.add_argument('--num-spatial-views', type=int, default=method_config["num-spatial-views"], help="Number of spatial crops to sample from each video during training")
    p.add_argument('--use-handcrafted-features', action='store_true', help="Use handcrafted features", default=method_config["use-handcrafted-features"])
    p.add_argument('--no-handcrafted-features', dest='use_handcrafted_features', action='store_false')
    p.add_argument('--keep-vision-prompts-throughout', action='store_true', help="Keep vision prompts throughout the network", default=method_config["keep-vision-prompts-throughout"])
    p.add_argument('--no-keep-vision-prompts', dest='keep_vision_prompts_throughout', action='store_false')
    p.add_argument('--has-discriminative-conditioning', action='store_true', help="Use discriminative conditioning", default=method_config["has-discriminative-conditioning"])
    p.add_argument('--no-discriminative-conditioning', dest='has_discriminative_conditioning', action='store_false')
    p.add_argument('--activation-clamp-value', type=float, default=method_config["activation-clamp-value"], help="Value to clamp activations to prevent overflow")
    p.add_argument('--videomix-type', type=str, default=method_config["videomix-type"], help="Type of VideoMix augmentation to use")
    p.add_argument('--videomix-prob', type=float, default=method_config["videomix-prob"], help="Probability of applying VideoMix augmentation")
    p.add_argument('--use-augmentation', action='store_true', help="Use data augmentation", default=method_config["use-augmentation"])
    p.add_argument('--no-augmentation', dest='use_augmentation', action='store_false')    
    p.add_argument('--fp16', action='store_true', help="Use mixed precision training", default=method_config["fp16"])
    p.add_argument('--no-fp16', dest='fp16', action='store_false')
    p.add_argument('--use-checkpointing', action='store_true', help="Use gradient checkpointing to save memory", default=method_config["use-checkpointing"])
    p.add_argument('--no-checkpointing', dest='use_checkpointing', action='store_false')
    p.add_argument('--grad-norm-max', type=float, default=method_config["grad-norm-max"], help="Maximum gradient norm for clipping")
    p.add_argument('--debug', action='store_true', help="Enable debug mode with more verbose logging", default=method_config["debug"])
    p.add_argument('--no-debug', dest='debug', action='store_false')
    p.add_argument('--epochs-to-skip', type=int, default=method_config["epochs-to-skip"], help="Number of epochs to skip")
    p.add_argument('--use-profiling', action='store_true', help="Enable profiling for performance analysis", default=method_config.get("use-profiling", False))
    p.add_argument('--no-profiling', dest='use_profiling', action='store_false')
    p.add_argument('--log-every-n-steps', type=int, default=method_config["log-every-n-steps"], help="Log to W&B every n steps")
    p.add_argument('--learn-just-pooling', action='store_true', help="Learn only the temporal pooling layer, keep other prompt parameters frozen", default=method_config.get("learn-just-pooling", False))
    p.add_argument('--no-learn-just-pooling', dest='learn_just_pooling', action='store_false')
    p.add_argument('--use-fresh-lr-scheduler', action='store_true', help="When loading from checkpoint, re-initialize the learning rate scheduler instead of loading its state", default=method_config.get("use-fresh-lr-scheduler", False))
    p.add_argument('--no-fresh-lr-scheduler', dest='use_fresh_lr_scheduler', action='store_false')

    
    method_name = args.prompt_optimisation_method.lower()
    if method_name in method_names[1:]:
        p.add_argument('--ctx-len-video', type=int, default=method_config["ctx-len-video"])

    if method_name == "vilt":
        p.add_argument('--num-tokens-per-layer-text', type=int, default=method_config["num-tokens-per-layer-text"])
        p.add_argument('--num-tokens-per-layer-vision', type=int, default=method_config["num-tokens-per-layer-vision"])

    if method_name == "vita":
        p.add_argument('--num-frames', type=int, default=method_config["num-frames"])
        p.add_argument('--num-frame-tokens', type=int, default=method_config["num-frame-tokens"])
        p.add_argument('--num-heads-summary-attention', type=int, default=method_config["num-heads-summary-attention"])
    
    args = p.parse_args()
    # Ensure that all keys in the dictionary use hyphens
    replace_underscore = {key.replace("_", "-"): value for key, value in vars(args).items()}
    args_dataset_config = {key: value for key, value in replace_underscore.items() if key in method_config["dataset-config"]}
    method_config["dataset-config"].update(args_dataset_config)
    method_config.update(replace_underscore)
    return args, method_config


def get_model_class(method: str):
    method = method.lower()
    if method == "video_coop":
        from ..modeling.CoOp import CoOpModel
        return CoOpModel
    elif method == "vidop":
        from ..modeling.VidOp import VidOpModel
        return VidOpModel
    elif method == "dual_coop":
        from ..modeling.Dual import DualModel
        return DualModel
    elif method == "vilt":
        from ..modeling.ViLT import ViLTModel
        return ViLTModel
    elif method == "vita":
        from ..modeling.ViTa import ViTaModel
        return ViTaModel
    elif method == "stt":
        from ..modeling.STT import STTModel
        return STTModel
    else:
        raise ValueError(f"Unknown method: {method}, should be one of {method_names}")

def load_checkpoint_given_trainer(trainer: CoOpModel, path_or_keys: str):
    from .paths import CHECKPOINT_JSON_PATH
    import json
    import os

    if os.path.exists(path_or_keys):
        trainer.load_context(path_or_keys)
        print(f"Loaded checkpoint from {path_or_keys}")
        return True
    else:
        if not os.path.exists(CHECKPOINT_JSON_PATH):
            raise FileNotFoundError(f"Checkpoint JSON file not found at {CHECKPOINT_JSON_PATH}. Please create it to map keys to checkpoint paths.")
        with open(CHECKPOINT_JSON_PATH, 'r') as f:
            checkpoint_dict = json.load(f)
        try:
            k1, k2, k3 = path_or_keys.split('/')
            checkpoint_path = checkpoint_dict[k1][k2][k3]
            trainer.load_context(checkpoint_path)
            print(f"Loading checkpoint from {checkpoint_path} for key '{path_or_keys}'")
            return True
        except KeyError:
            print(f"Key '{path_or_keys}' not found in checkpoint JSON. Starting from scratch")
    
    return False

def load_checkpoint_given_path(
        method_name, 
        checkpoint_path,
        method_config, 
        device, 
        class_names,
        wandb_logger = None,
        ):
    model_class = get_model_class(method_name)
    
    model = model_class(
        config=method_config,
        class_names=class_names,
        device=device,
        eval_class_names=class_names,
        wandb_logger=wandb_logger,
    )

    load_checkpoint_given_trainer(model, checkpoint_path)

    return model

def load_model_from_keys(keys: str, class_names: list[str], device: str = "cuda" if torch.cuda.is_available() else "cpu", wandb_logger = None):
    """
    Load a model given keys in the checkpoint JSON file.
    """
    method_config, checkpoint_path = get_config_and_path_from_keys(keys)
    method_name = keys.split('/')[0]
    model = load_checkpoint_given_path(
        method_name=method_name,
        checkpoint_path=checkpoint_path,
        method_config=method_config,
        device=device,
        class_names=class_names,
        wandb_logger=wandb_logger,
    )
    print(f"Loaded model '{method_name}' from checkpoint with keys '{keys}'")
    return model

def get_config_and_path_from_keys(keys: str):
    """
    Return model configuration and checkpoint path given keys in the checkpoint JSON file.
    """
    from .paths import CHECKPOINT_JSON_PATH
    from .read_config import read_method_config, read_dataset_config
    import json
    import os

    base_config = read_method_config(keys.split('/')[0])
    base_dataset_config = read_dataset_config("base")

    if not os.path.exists(CHECKPOINT_JSON_PATH):
        raise FileNotFoundError(f"Checkpoint JSON file not found at {CHECKPOINT_JSON_PATH}. Please create it to map keys to checkpoint paths.")
    with open(CHECKPOINT_JSON_PATH, 'r') as f:
        checkpoint_dict = json.load(f)
    try:
        k1, k2, k3 = keys.split('/')
        checkpoint_path = checkpoint_dict[k1][k2][k3]
        print(f"Loading checkpoint from {checkpoint_path} for key '{keys}'")
    except KeyError:
        raise KeyError(f"Key '{keys}' not found in checkpoint JSON.")

    config_path = os.path.join(os.path.dirname(checkpoint_path), 'config.json')
    with open(config_path, 'r') as f:
        method_config = json.load(f)
    # Add any values not in the saved config from the base configs (backwards compatibility)
    base_config.update(method_config)
    base_dataset_config.update(method_config["dataset-config"])
    base_config["dataset-config"] = base_dataset_config
    return base_config, checkpoint_path

def load_model_and_config_from_keys(model_name: str, dataset_train: str, run_id: str, class_names: list[str]):
    checkpoint_path = get_checkpoint_path(model_name, dataset_train, run_id)
    dirname = os.path.dirname(checkpoint_path)
    config_path = os.path.join(dirname, "config.json")

    method_config = read_config_from_file(config_path)

    model = load_checkpoint_given_path(
        method_name=model_name,
        checkpoint_path=checkpoint_path,
        method_config=method_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        class_names=class_names,
    )
    return model, method_config