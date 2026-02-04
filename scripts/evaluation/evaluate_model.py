if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Find project root
    project_root = Path(__file__).resolve().parent
    while not (project_root / "README.md").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))

    from src.eval.evaluation import run_evaluation

    key = "vidop/ucf101/best-no-hand-features-attention-0",
        

    model_name, dataset_train, run_id = key.split("/")
    eval_config = {
        "model_name": model_name,
        "dataset_train": dataset_train,
        "run_id": run_id,
        "dataset_test": "ucf101",
        "K-test": None, # Set to None to use the full dataset
        "num-classes": None, # Set to None to use all classes
        "class-protocol": "all",
        "use-handcrafted-prompts": False,
        "debug": False,
        "split": "test",
        "use-augmentation": True,
    }

    meta_config = {
        "batch-size": 16,
        "num-workers": 8,
        "use-wandb": True,
        "re-run": True
    }
    try:
        run_evaluation(eval_config, meta_config)
    except Exception as e:
        print(f"Error during evaluation of {key}: {e}")
