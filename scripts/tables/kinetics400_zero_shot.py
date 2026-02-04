#!/usr/bin/env python3
"""
Generate LaTeX table for Kinetics400 zero-shot transfer results (tab:kinetics400_zero-shot).
Zero-shot transfer from UCF101 to Kinetics400.
Note: The caption in the LaTeX says "HMDB51" but table label is "kinetics400_zero-shot" - keeping as is.
"""

import os
import sys

if __name__ == "__main__":
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

    # Test accuracies on Kinetics400 zero-shot
    results = {
        "VidOp (attention 0)": 9.0,
        "CoOp (mean)": 15.6,
        "CoOp (attention 0)": 5.0,
        "CoOp (attention 1)": 10.7,
        "CoOp (attention 2)": 9.9,
        "Dual (mean)": 19.5,
        "Dual (attention 0)": 3.7,
        "Dual-VideoMix (attention 0)": 3.9,
        "Dual (attention 1)": 13.1,
        "Dual (attention 2)": 12.2,
        "ViLt (mean)": 7.7,
        "Vita (mean)": 2.7,
        "STT (mean)": 10.5,
    }

    # Print LaTeX table
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{lSS}")
    print(r"\toprule")
    print(r"Model name & {Test (\%)} \\")
    print(r"\midrule")
    
    for model, test_acc in results.items():
        print(f"{model:30s} & {test_acc:.1f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Accuracies on HMDB51 in a zero-shot setting using models trained on UCF101.}")
    print(r"\label{tab:kinetics400_zero-shot}")
    print(r"\end{table}")
