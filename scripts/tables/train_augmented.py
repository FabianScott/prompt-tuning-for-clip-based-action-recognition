
import os
import sys

if __name__ == "__main__":
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

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