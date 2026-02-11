#!/usr/bin/env python3
"""
Generate LaTeX table for main UCF101 training results (tab:ucf101_results).
Train, validation and test accuracies with training times.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Find project root - scripts/tables/ is 2 levels deep
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    # Data: [train_acc, val_acc, test_acc, train_hours]
    results = {
        "VidOp (mean)": [52.1, 54.4, 51.4, 4],
        "VidOp H (mean)": [63.9, 52.5, 51.4, 4],
        "VidOp (attention 0)": [87.9, 84.3, 82.9, 7],
        "VidOp H (attention 0)": [81.6, 58.0, 60.7, 5],
        "CoOp (mean)": [88.6, 86.8, 85.6, 6.5],
        "CoOp (attention 0)": [99.7, 93.3, 91.9, 2],
        "CoOp (attention 1)": [100.0, 93.5, 92.5, 8],
        "CoOp (attention 2)": [100.0, 92.6, 91.7, 8],
        "Dual (mean)": [95.2, 93.3, 91.7, 8.5],
        "Dual (attention 0)": [100.0, 93.2, 91.3, 7],
        "Dual (attention 1)": [100.0, 95.0, 93.2, 10],
        "Dual (attention 2)": [97.6, 95.0, 90.7, 11],
        "ViLt (mean)": [93.6, 86.5, 84.8, 18],
        "Vita (mean)": [99.3, 91.0, 90.8, 15],
        "STT (mean)": [99.9, 94.9, 92.0, 14],
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
        if isinstance(hours, int):
            hours_str = str(hours)
        else:
            hours_str = f"{hours:.1f}"
        print(f"{model:30s} & {train:.1f} & {val:.1f} & {test:.1f} & {hours_str} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Train, validation and test accuracies on UCF101. Parentheses denote temporal pooling method. Attention 0 is learned from scratch; attention 1 is added after training and learned alone; attention 2 adds attention after training and fine-tunes the full model. Train times are rounded to the nearest half hour.}")
    print(r"\label{tab:ucf101_results}")
    print(r"\end{table}")
