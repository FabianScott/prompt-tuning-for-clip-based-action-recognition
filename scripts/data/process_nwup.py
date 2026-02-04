if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Find project root
    project_root = Path(__file__).resolve().parent
    while not (project_root / "README.md").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))
    print(f"Project root: {project_root}")
    from src.data.nwpu_loading import NWUPDataset, create_pytorch_dataset_folder, generate_action_detection_annotations

    annotations = generate_action_detection_annotations()

    create_pytorch_dataset_folder()

    dataset = NWUPDataset()

    print(f"Dataset has {len(dataset)} samples and {len(dataset.class_names)} classes.")

    