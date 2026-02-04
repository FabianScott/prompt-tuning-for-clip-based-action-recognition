if __name__ == "__main__":
    import os
    import sys
    import traceback

    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

    from src.eval.evaluation import run_evaluation

    keys_list = [
        # "video_coop/hmdb51/k-16-attention-0",
        # "video_coop/hmdb51/k-16-ucf-attention-1",
        # "dual_coop/kinetics400/k4-attention-0",
        # "dual_coop/kinetics400/k-4-ucf-attention-1",
        # "video_coop/kinetics400/full-ep-8",
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

    settings_to_vary = {
        # "dataset_test": "kinetics400",
        "dataset_test": "ucf101",
    }

    for key in keys_list:
        for setting_key in settings_to_vary.keys():
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
                "use-augmentation": False,
                "remove-black-strips": True,
                "num-temporal-views": 1,
            }
            eval_config[setting_key] = settings_to_vary[setting_key]
            print(f"Running evaluation for {key} with {setting_key}={eval_config[setting_key]}={settings_to_vary[setting_key]}")

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
                traceback.print_exc()
