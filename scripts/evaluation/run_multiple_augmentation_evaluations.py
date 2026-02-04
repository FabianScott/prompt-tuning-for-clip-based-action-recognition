if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Find project root
    project_root = Path(__file__).resolve().parent
    while not (project_root / "README.md").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))

    from src.eval.evaluation import run_evaluation

    keys_list = [
        # "video_coop/hmdb51/k-16-attention-0",
        # "video_coop/hmdb51/k-16-ucf-attention-1",
        # "dual_coop/kinetics400/k4-attention-0",
        # "dual_coop/kinetics400/k-4-ucf-attention-1",
        # "vidop/ucf101/best-no-hand-features-attention-0",
        # "video_coop/ucf101/best-mean",
        # "video_coop/ucf101/best-attention-0",
        # "video_coop/ucf101/best-attention-1",
        # "video_coop/ucf101/best-attention-2",
        # "dual_coop/ucf101/best-mean",
        # "dual_coop/ucf101/best-attention-0",
        # "dual_coop/ucf101/videomix-attention-0",
        # "dual_coop/ucf101/best-attention-1",
        # "dual_coop/ucf101/best-attention-2",
        # "vilt/ucf101/best-mean",
        # "vita/ucf101/ep-16",
        # "stt/ucf101/best-true-val"
        "vilt/ucf101/videomix",
        "vita/ucf101/videomix",
        "stt/ucf101/videomix"
    ]
    aug_run = True

    augmentation_settings = [
        ("random-horizontal-flip", {
            "p": 1.0
            }),
        ("color-jitter", {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.1,
            }),
        ("gaussian-blur", {
            "kernel-size": 5,
            "sigma": [1, 1],
            }),
        ("salt-and-pepper-noise", {
            "probability": 0.05,
            }),
        ("cutout", {
            "p": 0.1, 
            "scale": [0.02, 0.33], 
            "ratio": [0.3, 3.3]
            }),
        ("cutout", {
            "p": 1.0, 
            "scale": [0.02, 0.33], 
            "ratio": [0.3, 3.3]
            }),

    ] if aug_run else [(None, None)]

    for key in keys_list:
        for aug_name, aug_cfg in augmentation_settings:
            model_name, dataset_train, run_id = key.split("/")

            eval_config = {
                "model_name": model_name,
                "dataset_train": dataset_train,
                "run_id": run_id,
                "dataset_test": "ucf101",
                "K-test": None,
                "num-classes": None,
                "class-protocol": "all",
                "use-handcrafted-prompts": False,
                "debug": False,
                "split": "test",
                "use-augmentation": aug_run,
                "num-temporal-views": 4,
            }

            if aug_name is not None:
                eval_config[aug_name] = aug_cfg

            meta_config = {
                "batch-size": 16,
                "num-workers": 8,
                "use-wandb": True,
                "re-run": True,
            }

            try:
                run_evaluation(eval_config, meta_config)
            except Exception as e:
                print(f"Error during evaluation of {key}: {e}")
