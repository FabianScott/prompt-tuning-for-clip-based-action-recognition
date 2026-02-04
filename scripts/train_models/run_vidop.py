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
        "temporal-pooling": "attention",
        "use-handcrafted-features": False,
        "use-augmentation": True,
    }
    
    train_model(
        method_default="vidop",
        dataset_default="ucf101",
        fixed_config=fixed_config
    )
