
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
        "VidOp (attention 0)": [77.0],
        "CoOp (mean)": [85.0],
        "CoOp (attention 0)": [90.6],
        "CoOp (attention 1)": [91.6],
        "CoOp (attention 2)": [91.1],
        "Dual (mean)": [80.5],
        "Dual (attention 0)": [89.3],
        "Dual-VideoMix (attention 0)": [84.4],
        "Dual (attention 1)": [82.1],
        "Dual (attention 2)": [72.1],
        "ViLt": [80.9],
        "ViTa": [89.4],
        "STT": [90.1],
    }
    kinds = ["test"]
    print_augmented_results(
        augmented_train, 
        kinds=kinds
        )