if __name__ == '__main__':
    import os
    import sys
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    print(f"Current working directory: {os.getcwd()}")
    sys.path.append(os.getcwd())
    from src.data.nwpu_loading import NWUPDataset, create_pytorch_dataset_folder, generate_action_detection_annotations

    annotations = generate_action_detection_annotations()

    create_pytorch_dataset_folder()

    dataset = NWUPDataset()

    print(f"Dataset has {len(dataset)} samples and {len(dataset.class_names)} classes.")

    