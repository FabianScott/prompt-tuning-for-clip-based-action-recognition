# Requires pip install wandb[media]

import os
import sys
import wandb
import torch
from tqdm import tqdm
from typing import Optional

def run_explainer(grad_cam_config:dict, run_name:str):
    from transformers import CLIPProcessor

    from src.modeling.explainer import VideoExplainer
    from src.data.dataset_builders import build_dataset
    from src.configs.parse_run_args import load_checkpoint_given_path
    from src.configs.read_config import get_checkpoint_path
    from src.data.dataset_builders import build_dataset, build_balanced_subset
    from src.data.dataloading import build_dataloader
    from src.configs.read_token import read_token
    from src.configs.read_config import read_method_config, read_config_from_file, read_dataset_config
    from src.configs.paths import CLIP_MODEL_CACHE_DIR

    classes_to_map = grad_cam_config["classes-to-map"]
    classes_to_mix = grad_cam_config.get("classes-to-mix", None)  # If using VideoMix

    checkpoint_path = get_checkpoint_path(
        method_name=grad_cam_config["model-name"],
        dataset_name=grad_cam_config["dataset-train"],
        run_id=grad_cam_config["run-id"],
    )

    dirname = os.path.dirname(checkpoint_path)
    config_path = os.path.join(dirname, "config.json")

    method_config = read_config_from_file(config_path)
    base_config = read_method_config(grad_cam_config["model-name"])
    base_dataset_config = read_dataset_config(grad_cam_config["dataset-test"])
    # Using the base configs to fill in missing keys
    method_config["dataset-config"].update(base_dataset_config)
    for key in grad_cam_config:
        if key in method_config["dataset-config"]:
            method_config["dataset-config"][key] = grad_cam_config[key]
        elif key in method_config:
            method_config[key] = grad_cam_config[key]
        elif key not in method_config and key in base_config:
            method_config[key] = grad_cam_config[key]

    base_config.update(method_config)
    method_config = base_config
    grad_cam_config["train-config"] = method_config

    tmp_processor = CLIPProcessor.from_pretrained(method_config["clip-model"], cache_dir=CLIP_MODEL_CACHE_DIR)
    dataset = build_dataset(
        name=grad_cam_config["dataset-test"],
        split=grad_cam_config["split"],
        processor=tmp_processor,
        config=method_config
    )
    if classes_to_mix is not None:
        from src.data.videomix import VideoMixDataset
        dataset = VideoMixDataset(dataset, config=method_config)

    class_names = dataset.class_names

    if grad_cam_config["use-wandb"]:
        wandb.login(key=read_token('tokens/wandb.txt', 'Weights & Biases'))
    wandb_logger = wandb.init(
        project="Thesis - Video Classification",
        config=grad_cam_config,
        name="GRADCAM-" + run_name,
        tags=["grad-cam"],
        ) if grad_cam_config["use-wandb"] else None

    model = load_checkpoint_given_path(
        method_name=grad_cam_config["model-name"],
        checkpoint_path=checkpoint_path,
        method_config=method_config,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        class_names=class_names,
        wandb_logger=wandb_logger,
    )

    target_layer = model.clip.vision_model.encoder.layers[-1].layer_norm2
    video_explainer = VideoExplainer(
        model=model, 
        target_layer=target_layer
        )

    for i, class_name in enumerate(classes_to_map):
        if classes_to_mix is not None:
            mix_class_name = classes_to_mix[i]
            mix_video_index = find_video_index(dataset, mix_class_name, n_skip=1)
            if mix_video_index is None:
                print(f"Could not find a second video for mixing with class {mix_class_name}. Skipping VideoMix for class {class_name}.")
            else:
                dataset.set_idx_to_mix(mix_video_index)
        
        video_index = find_video_index(dataset, class_name, n_skip=grad_cam_config["video-index"])
        if video_index is None:
            print(f"Could not find video for class {class_name}. Skipping.")
            continue
        dat_out = dataset[video_index]
        if dat_out is None:
            print(f"Data output is None for index {video_index}. Skipping.")
            continue
        video, _ = dat_out
        videos = video.unsqueeze(0)  # (1, V, T, C, H, W)
        video_explainer.explain(
            videos=videos,
            text_features=model.encode_texts(class_names) if grad_cam_config["method"] == "gradcam" else None,
            class_idx=dataset.class_to_idx[class_name],
            method=grad_cam_config["method"],
            fps=1,
            log_to_wandb=grad_cam_config["use-wandb"],
            plot_path=f"figures/explainer/{run_name}/{class_name}/{video_index}-tmp_view.png"
        )

    return 


def find_video_index(dataset, target_class_name, n_skip: int = 0) -> Optional[int]:
    target = dataset.class_to_idx[target_class_name]
    n_skipped = 0

    for idx, (_, label) in enumerate(tqdm(dataset.samples, desc=f"Finding video for class {target_class_name}")):
        if label == target:
            if n_skipped < n_skip:
                n_skipped += 1
                continue
            return idx

    print(f"Class {target_class_name} not found in dataset.")
    return None


if __name__ == "__main__":
    from pathlib import Path
    
    # Find project root
    project_root = Path(__file__).resolve().parent
    while not (project_root / "README.md").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))

    # from src.modeling.grad_cam import run_grad_cam
    keys_list = [
        # "video_coop/hmdb51/k-16-attention-0",
        # "video_coop/hmdb51/k-16-ucf-attention-1",
        # "dual_coop/kinetics400/k4-attention-0",
        # "dual_coop/kinetics400/k-4-ucf-attention-1",
        "vidop/ucf101/best-no-hand-features-attention-0",
        "video_coop/ucf101/best-mean",
        "video_coop/ucf101/best-attention-0",
        "video_coop/ucf101/best-attention-1",
        "video_coop/ucf101/best-attention-2",
        "dual_coop/ucf101/best-mean",
        "dual_coop/ucf101/best-attention-0",
        "dual_coop/ucf101/videomix-attention-0",
        "dual_coop/ucf101/best-attention-1",
        "dual_coop/ucf101/best-attention-2",
        "vilt/ucf101/best-mean",
        "vita/ucf101/ep-16",
        "stt/ucf101/best-true-val"
    ]

    actions = [
        "ApplyEyeMakeup",
        "BabyCrawling",
        "BasketballDunk",
        "Biking",
        "Billiards",
        "BlowingCandles",
        "Bowling",
        "BoxingPunchingBag",
        "CuttingInKitchen",
        "Diving",
        "Fencing",
        "FrisbeeCatch",
        "HorseRace",
        "HorseRiding",
        "IceDancing",
        "Kayaking",
        "Knitting",
        "MilitaryParade",
        "PlayingCello",
        "PlayingDaf",
        "PlayingDhol",
        "PlayingFlute",
        "PlayingGuitar",
        "PlayingPiano",
        "PlayingSitar",
        "PlayingTabla",
        "PlayingViolin",
        "RockClimbingIndoor",
        "Rowing",
        "SalsaSpin",
        "Skijet",
        "SkyDiving",
        "SoccerPenalty",
        "SumoWrestling",
        "TableTennisShot",
        "VolleyballSpiking",
        "WritingOnBoard",
    ]


    for key in tqdm(keys_list, desc="Running Explainer for all models"):
        model_name, dataset_train, run_id = key.split("/")
        
        explainer_config = {
            "model-name": model_name,
            "dataset-train": dataset_train,
            "run-id": run_id,
            "dataset-test": "ucf101",
            "num-classes": 101,
            "class-protocol": "all",
            "debug": False,
            "split": "test",
            "video-index": 1,  # Index of the video in the dataset to analyze,
            "use-wandb": False,
            "classes-to-map": actions,
            "classes-to-mix": None, #["Basketball",],  # If using VideoMix,
            "num-spatial-views": 1,     # Just take the original video to simplify,
            "method": "attention-rollout-cls-weighted",  # "grad-cam" or "attention-rollout"
            "use-augmentation": False,  # Should always be False for explainer runs
        }

        run_explainer(
            explainer_config, 
            run_name=f"{explainer_config['model-name']}"
                    f"{explainer_config['run-id']}-"
                    f"{explainer_config['dataset-test']}-"
                    f"{explainer_config['video-index']}-"
                    f"{explainer_config['method']}"
                    )