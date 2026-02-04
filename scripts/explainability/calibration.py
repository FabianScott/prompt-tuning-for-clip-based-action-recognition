import sys
from pathlib import Path

if __name__ == "__main__":
    # Find project root
    project_root = Path(__file__).resolve().parent
    while not (project_root / "README.md").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))

    from src.modeling.calibration import run_calibration_evaluation
    from transformers import CLIPProcessor
    from src.configs.paths import HPC_SCRATCH_PATH
    from src.data.dataset_builders import build_dataset
    
    model_keys = [
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
        "stt/ucf101/best-true-val",
        "vilt/ucf101/videomix",
        "vita/ucf101/videomix",
        "stt/ucf101/videomix",
    ]


    meta_config = {
        "batch-size": 16,
        "num-workers": 8,
        "use-wandb": True,
        "re-run": True,
    }

    for model_key in model_keys:
        model_name, dataset_name, run_id = model_key.split("/")
        eval_config = {
            "model_name": model_name,
            "dataset-train": dataset_name,
            "dataset-test": "ucf101",
            "split": "test",
            "run_id": model_key.replace("/", "-"),
            "use-augmentation": False,
            "use-handcrafted-prompts": False,
            "K-test": None,
        }
        run_calibration_evaluation(
            eval_config=eval_config,
            meta_config=meta_config,
            model_key=model_key,
        )
