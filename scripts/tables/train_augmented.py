
import sys
from pathlib import Path

if __name__ == "__main__":
    # Find project root
    project_root = Path(__file__).resolve().parent
    while not (project_root / "README.md").exists() and project_root.parent != project_root:
        project_root = project_root.parent
    sys.path.insert(0, str(project_root))

    from src.tables.augmentation_utils import print_augmented_results
    augmented_train = {
        "VidOp (attention 0)": [81.0, 75.3, 72.7, 7],
        "CoOp (mean)":        [78.0, 76.6, 85.6, 8],
        "CoOp (attention 0)": [94.4, 82.2, 77.5, 4],
        "Dual (mean)":        [80.8, 84.0, 76.4, 6.5],
        "Dual (attention 0)": [94.4, 82.2, 84.0, 7],
        "ViLt":               [93.6, 86.5, 76.7, 21],
        "ViTa":               [92.9, 82.8, 81.4, 13],
        "STT":                [98.7, 85.8, 82.9, 21],
    }

    kinds = ["train", "val", "test", "other"]
    print_augmented_results(
        augmented_train, 
        kinds=kinds
        )