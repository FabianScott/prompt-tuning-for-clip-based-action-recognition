if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    from src.modeling.train import train_model
    
    # Dataset configuration for training
    dataset_config = {
        "batch-size": 4,
        "num-workers": 4,
        "batches-per-backprop": 32,
        "K-train": None,
        "K-val": 4,
    }

    # Training configuration
    fixed_config = {
        "dataset-config": dataset_config,
        "continue-from": "video_coop/kinetics400/full-ep-6-16-tokens",
        "use-augmentation": False,
        "temporal-pooling": "attention",
        "ctx-len": 16,
    }
    
    train_model(
        method_default="video_coop",
        dataset_default="kinetics400",
        fixed_config=fixed_config
    )
