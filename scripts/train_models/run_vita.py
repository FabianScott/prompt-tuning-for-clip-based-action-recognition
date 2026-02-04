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

    # Training configuration
    fixed_config = {
        "continue-from": None,
        "use-augmentation": False,
        "temporal-pooling": "mean",
        "videomix-type": "spatial",
        "dataset-config": {
            "K-train": None,
            "batch-size": 4,
            "num-workers": 2,
            "batches-per-backprop": 32,
        },
    }

    train_model(
        method_default="vita",
        dataset_default="ucf101",
        fixed_config=fixed_config
    )