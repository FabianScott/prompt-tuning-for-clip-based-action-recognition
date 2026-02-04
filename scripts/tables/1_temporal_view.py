
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
        "VidOp (attention 0)": [81.8],
        "CoOp (mean)":        [85.3],
        "CoOp (attention 0)": [90.0],
        "CoOp (attention 1)": [91.0],
        "CoOp (attention 2)": [89.5],
        "Dual (mean)":        [91.8],
        "Dual (attention 0)": [90.7],
        "Dual (attention 1)": [88.9],
        "Dual (attention 2)": [92.3],
        "ViLt":        [83.7],
        "ViTa":        [89.5],
        "STT":         [90.0],
        "Dual-VideoMix (attention 0)": [88.8],
        "ViLt-VideoMix": [82.0],
        "ViTa-VideoMix": [86.7],
        "STT-VideoMix":  [89.2],
    }

    kinds = ["test"]
    print_augmented_results(
        augmented_train, 
        kinds=kinds
        )