#!/usr/bin/env python3
"""
Generate LaTeX table for HMDB51 K-shot learning results (tab:hmdb51_kshot).
Both 4-shot and 16-shot results on HMDB51.
"""

import os
import sys

if __name__ == "__main__":
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

    # Data: [16-shot-val, 16-shot-test, 4-shot-val, 4-shot-test]
    results = {
        "CoOp (attention 0)": [63.2, 67.6, 24.7, 21.7],
        "CoOp T (attention 1)": [69.0, 67.6, 27.8, 28.9],
        "Dual (attention 0)": [68.4, 60.6, 46.9, 43.5],
        "Dual T (attention 1)": [72.1, 61.0, 30.5, 28.7],
        "ViLT T": [42.6, 42.1, 23.2, 26.3],
        "Vita T": [62.6, 54.6, 29.4, 27.4],
    }

    # Print LaTeX table
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\begin{tabular}{lSSSS}")
    print(r"\toprule")
    print(r"Model name & {16-shot (val)} & {16-shot (test)} & {4-shot (val)} & {4-shot (test)} \\")
    print(r"\midrule")
    
    for model, values in results.items():
        val_16, test_16, val_4, test_4 = values
        print(f"{model:30s} & {val_16:.1f} & {test_16:.1f} & {val_4:.1f} & {test_4:.1f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"""\caption{HMDB51 K-shot results. Parenthesis states the attention pooling method and a "T" means this model was transferred from UCF101.}""")
    print(r"\label{tab:hmdb51_kshot}")
    print(r"\end{table}")
