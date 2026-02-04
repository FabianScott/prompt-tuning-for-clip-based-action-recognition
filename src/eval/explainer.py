# Requires pip install wandb[media]

import os
import sys
import torch
from tqdm import tqdm
from typing import Optional

def run_explainer(grad_cam_config:dict, run_name:str):
    from ..modeling.explainer import VideoExplainer
    from ..configs.parse_run_args import load_checkpoint_given_path
    from ..configs.read_config import read_method_config
    from .eval_utils import (
        load_config_and_checkpoint,
        prepare_method_config,
        build_evaluation_dataset,
        init_wandb_logger
    )

    classes_to_map = grad_cam_config["classes-to-map"]
    classes_to_mix = grad_cam_config.get("classes-to-mix", None)  # If using VideoMix

    # Load config and checkpoint
    method_config, checkpoint_path = load_config_and_checkpoint(grad_cam_config)
    
    # Prepare method config
    base_config = read_method_config(grad_cam_config["model-name"])
    method_config = prepare_method_config(method_config, grad_cam_config, base_config)
    grad_cam_config["train-config"] = method_config

    # Build dataset
    dataset, class_names = build_evaluation_dataset(grad_cam_config, method_config)
    
    if classes_to_mix is not None:
        from ..data.videomix import VideoMixDataset
        dataset = VideoMixDataset(dataset, config=method_config)
        class_names = dataset.class_names

    # Initialize wandb
    wandb_logger = init_wandb_logger(
        meta_config={"use-wandb": grad_cam_config.get("use-wandb", False)},
        method_config=grad_cam_config,
        test_name="GRADCAM-" + run_name,
        tags=["grad-cam"]
    )

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
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

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
            "model_name": model_name,  # Using standard key name
            "dataset_train": dataset_train,  # Using standard key name
            "run_id": run_id,  # Using standard key name
            "model-name": model_name,  # Keep for backward compatibility
            "dataset-train": dataset_train,  # Keep for backward compatibility
            "run-id": run_id,  # Keep for backward compatibility
            "dataset_test": "ucf101",  # Using standard key name
            "dataset-test": "ucf101",  # Keep for backward compatibility
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