#!/usr/bin/env python3
"""
Generate LaTeX table for UCF101 calibration metrics (tab:ucf101_calibration).
Includes NLL, ECE, Adaptive ECE, Classwise ECE, MCE, and Brier Score.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Find project root - scripts/tables/ is 2 levels deep
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    from src.tables.results_data import UCF101_CALIBRATION_METRICS
    # Print LaTeX table
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{lSSSSSSl}")
    print(r"\toprule")
    print(r"Model & {NLL} & {ECE} & {Adaptive ECE} & {Classwise ECE} & {MCE} & {Brier Score} \\")
    print(r"\midrule")
    
    for model, metrics in UCF101_CALIBRATION_METRICS.items():
        nll, ece, adaptive_ece, classwise_ece, mce, brier_score = metrics
        model = model.replace("VideoMix", "V").replace("attention ", "a-")
        print(f"{model} & {nll:.2f} & {ece:.2f} & {adaptive_ece:.2f} & {classwise_ece:.2f} & {mce:.2f} & {brier_score:.2f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Model calibration metrics on UCF101 validation set. NLL: Negative Log-Likelihood, ECE: Expected Calibration Error, MCE: Maximum Calibration Error.}")
    print(r"\label{tab:ucf101_calibration}")
    print(r"\end{table}")
