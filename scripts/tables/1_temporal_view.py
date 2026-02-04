
import os
import sys

if __name__ == "__main__":
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

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