import json

from .datasets_and_methods import dataset_names, method_names
from .paths import CHECKPOINT_JSON_PATH

def read_config_from_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def read_method_config(method_name: str, src_path="src", use_augmentation=False) -> dict:
    if method_name not in method_names:
        raise ValueError(f"Unknown method name: {method_name}, should be on of {method_names}")
    base_config = read_config_from_file(f"{src_path}/configs/methods/base.json")
    config = read_config_from_file(f"{src_path}/configs/methods/{method_name}.json")
    base_config.update(config)
    if base_config["use-augmentation"] or use_augmentation:
        base_config["augmentation-dict"] = read_augmentation_config(src_path=src_path)
        base_config["use-augmentation"] = True
    return base_config

def read_dataset_config(dataset_name: str, src_path="src") -> dict:
    if dataset_name not in dataset_names:
        raise ValueError(f"Unknown dataset: {dataset_name}, should be one of {dataset_names}")
    base_config = read_config_from_file(f"{src_path}/configs/datasets/base.json")
    config = read_config_from_file(f"{src_path}/configs/datasets/{dataset_name}.json")
    base_config.update(config)
    return base_config

def read_augmentation_config(src_path="src") -> dict:
    aug_config = read_config_from_file(f"{src_path}/configs/datasets/augmentations-dict.json")
    return aug_config

def return_combined_config(model_name: str, dataset_name: str, src_path="src", use_augmentation=False) -> dict:
    """Load both dictionaries and put the dataset-config into the method config"""
    method_config = read_method_config(model_name, src_path=src_path, use_augmentation=use_augmentation)
    dataset_config = read_dataset_config(dataset_name, src_path=src_path)
    method_config["dataset-config"] = dataset_config
    return method_config

def save_config(file_path, config: dict, verbose=True):
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"Saved config file to {file_path}")

if __name__ == "__main__":
    for model_name in method_names:
        config = read_method_config(model_name)
        print(config)


def get_checkpoint_path(method_name, dataset_name, run_id):
    with open(CHECKPOINT_JSON_PATH, 'r') as f:
        checkpoint_dict = json.load(f)
        checkpoint_path = checkpoint_dict[method_name][dataset_name][run_id]
    return checkpoint_path