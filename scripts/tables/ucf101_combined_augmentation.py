#!/usr/bin/env python3
"""
Generate LaTeX table for UCF101 combined augmentation results (tab:ucf101_augmentation).
Test accuracies with cutout, salt and pepper noise, and random horizontal flipping applied.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Find project root - scripts/tables/ is 2 levels deep
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from src.tables.augmentation_utils import baseline_test

    # Test accuracies on augmented test set
    augmented_test = {
        "VidOp (attention 0)": 64.9,
        "CoOp (mean)": 69.2,
        "CoOp (attention 0)": 69.0,
        "CoOp (attention 1)": 72.2,
        "CoOp (attention 2)": 64.9,
        "Dual (mean)": 75.7,
        "Dual (attention 0)": 73.6,
        "Dual-VideoMix (attention 0)": 76.6,
        "Dual (attention 1)": 73.1,
        "Dual (attention 2)": 68.5,
        "ViLt (mean)": 47.9,
        "Vita (mean)": 52.8,
        "STT (mean)": 69.4,
    }

    # Print LaTeX table
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{lSS}")
    print(r"\toprule")
    print(r"Model name & {Test (\%)} & {$\Delta$ (new -- old)} \\")
    print(r"\midrule")
    
    for model, test_acc in augmented_test.items():
        baseline = baseline_test.get(model, None)
        if baseline is not None:
            delta = test_acc - baseline
            print(f"{model:30s} & {test_acc:.1f} & {delta:+.1f} \\\\")
        else:
            print(f"{model:30s} & {test_acc:.1f} & {{--}} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Accuracies on the augmented test set with cutout (p=0.1, ratio=(0.3,3.3) and scale=(0.02,0.33), random horizontal flip (p=0.5), salt noise (p=0.025) and pepper noise (p=0.025).}")
    print(r"\label{tab:ucf101_augmentation}")
    print(r"\end{table}")
