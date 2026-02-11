if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Find project root - scripts/tables/ is 2 levels deep
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from src.tables.augmentation_utils import print_augmented_results
    augmented_individually = {
        #  {H-Flip} & {Colour Jitter} & {Blur} & {S & P} & Cutout (p=0.1) Cutout (p=1)
        "VidOp (attention 0)": [82.2, 35.4, 78.7, 65.0, 82.4, 73.1],
        "CoOp (mean)":        [85.0, 44.4, 68.2, 82.4, 85.3, 77.5],
        "CoOp (attention 0)": [90.8, 40.2, 86.8, 67.5, 90.8, 78.2],
        "CoOp (attention 1)": [91.3, 42.7, 87.4, 70.9, 91.3, 81.6],
        "CoOp (attention 2)": [90.7, 40.3, 85.1, 64.1, 90.5, 80.2],
        "Dual (mean)":        [91.9, 48.5, 90.1, 74.9, 92.0, 82.6],
        "Dual (attention 0)": [91.2, 41.1, 87.9, 71.5, 91.1, 78.8],
        "Dual (attention 1)": [90.8, 47.9, 88.9, 73.6, 90.5, 79.4],
        "Dual (attention 2)": [92.7, 43.9, 90.2, 67.6, 92.7, 82.1],
        "ViLt":        [83.9, 32.3, 80.7, 46.7, 83.8, 69.0],
        "ViTa":        [89.6, 31.7, 79.1, 52.4, 89.6, 78.4],
        "STT":         [90.3, 38.8, 82.4, 70.0, 90.3, 81.0],
        "Dual-VideoMix (attention 0)": [90.1, 48.6, 85.2, 75.6, 90.1, 85.6],
        "ViLt-VideoMix": [83.0, 27.5, 80.0, 70.1, 83.1, 83.0],
        "ViTa-VideoMix": [86.6, 40.5, 84.2, 68.1, 86.7, 83.2],
        "STT-VideoMix": [90.7, 43.0, 84.6, 71.1, 90.5, 84.7],
    }
    kinds = ["test" for _ in augmented_individually["VidOp (attention 0)"]]
    
    # First table (first 3 columns)
    first_table = {k: v[:3] for k, v in augmented_individually.items()}
    print_augmented_results(first_table, kinds=kinds[:3])

    print("\n\n\n")

    # Second table (last 3 columns)
    second_table = {k: v[3:] for k, v in augmented_individually.items()}
    print_augmented_results(second_table, kinds=kinds[3:])