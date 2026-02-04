#!/usr/bin/env python3
"""
Generate LaTeX table for Kinetics400 K-shot learning results (tab:kinetics_kshot).
Both 4-shot and 16-shot results on Kinetics400.
"""

import os
import sys

if __name__ == "__main__":
    if os.getcwd().endswith("notebooks"):
        os.chdir("..")
    sys.path.append(os.getcwd())

    # Data: [16-shot-val, 16-shot-test, 4-shot-val, 4-shot-test]
    # None values represent empty cells ({})
    results = {
        "CoOp (attention 0)": [57.1, 54.6, 42.8, 23.1],
        "CoOp T (attention 1)": [61.7, 59.9, 50.4, 23.1],
        "Dual (attention 0)": [63.9, 60.3, 43.8, 42.0],
        "Dual T (attention 1)": [63.3, 61.9, 55.6, 58.7],
        "CoOp+Dual T Mixture": [None, 63.7, None, None],
        "STT T": [22.5, None, None, None],
        "Vita T": [13.2, 13.0, None, None],
        "ViLT T": [17.9, 17.4, None, None],
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
        
        val_16_str = "{}" if val_16 is None else f"{val_16:.1f}"
        test_16_str = "{}" if test_16 is None else f"{test_16:.1f}"
        val_4_str = "{}" if val_4 is None else f"{val_4:.1f}"
        test_4_str = "{}" if test_4 is None else f"{test_4:.1f}"
        
        print(f"{model:30s} & {val_16_str} & {test_16_str} & {val_4_str} & {test_4_str} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"""\caption{Kinetics400 K-shot results. Parenthesis states the attention pooling method and a "T" means this model was transferred from UCF101. STT performed so poorly after 48 hours of training that no further experiments were conducted.}""")
    print(r"\label{tab:kinetics_kshot}")
    print(r"\end{table}")
