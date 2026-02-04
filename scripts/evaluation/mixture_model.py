import os
import sys


if __name__ == "__main__":
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())
    from src.modeling.MixtureModel import build_mixture_from_sources, run_mixture_evaluation
    from transformers import CLIPProcessor
    from src.configs.paths import HPC_SCRATCH_PATH
    from src.data.dataset_builders import build_dataset
    
    sources = [
        # "dual_coop/ucf101/best-attention-1",
        # "video_coop/ucf101/best-attention-0"
        "video_coop/kinetics400/k-16-ucf-attention-1",
        "dual_coop/kinetics400/k-16-ucf-attention-1"
    ]
    eval_config = {
        "model_name": "coop",
        "dataset-train": "kinetics400",
        "dataset-test": "kinetics400",
        "split": "test",
        "run_id": "mixture_v1",
        "use-augmentation": False,
        "use-handcrafted-prompts": False,
        "K-test": None,
    }

    meta_config = {
        "batch-size": 16,
        "num-workers": 8,
        "use-wandb": True,
        "re-run": True,
    }

    run_mixture_evaluation(
        eval_config=eval_config,
        meta_config=meta_config,
        key_paths=sources,
        weights=None,
    )
