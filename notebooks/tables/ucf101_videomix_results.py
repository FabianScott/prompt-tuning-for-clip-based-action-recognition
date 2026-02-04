#!/usr/bin/env python3
"""
Generate LaTeX table for UCF101 VideoMix augmentation results (tab:ucf101_results_videomix).
Train, validation and test accuracies for models trained with VideoMix data augmentation.
"""

import os
import sys

if __name__ == "__main__":
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

    # Data: [train_acc, val_acc, test_acc, train_hours]
    # Note: train_acc is empty ({}) in the LaTeX
    results = {
        "Dual-VideoMix (attention 0)": [None, 92.3, 91.3, 7],
        "ViLt-VideoMix (mean)": [None, 87.0, 85.1, 18],
        "Vita-VideoMix (mean)": [None, 90.8, 86.8, 15],
        "STT-VideoMix (mean)": [None, 91.5, 91.5, 14],
    }

    # Print LaTeX table
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{lSSSl}")
    print(r"\toprule")
    print(r"Model name & {Train (\%)} & {Validation (\%)} & {Test (\%)} & {Train time (hours)} \\")
    print(r"\midrule")
    
    for model, values in results.items():
        train, val, test, hours = values
        train_str = "{}" if train is None else f"{train:.1f}"
        hours_str = str(hours) if isinstance(hours, int) else f"{hours:.1f}"
        print(f"{model:30s} & {train_str} & {val:.1f} & {test:.1f} & {hours_str} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Train, validation and test accuracies on UCF101 for models trained using the VideoMix data augmentation.}")
    print(r"\label{tab:ucf101_results_videomix}")
    print(r"\end{table}")
