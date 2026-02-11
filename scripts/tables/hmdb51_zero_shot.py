#!/usr/bin/env python3
"""
Generate LaTeX table for HMDB51 zero-shot transfer results (tab:hmdb51_zero-shot).
Zero-shot transfer from UCF101 to HMDB51.
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Find project root - scripts/tables/ is 2 levels deep
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    # Test accuracies on HMDB51 zero-shot
    results = {
        "VidOp (attention 0)": 19.6,
        "CoOp (mean)": 21.9,
        "CoOp (attention 0)": 10.2,
        "CoOp (attention 1)": 25.2,
        "CoOp (attention 2)": 24.4,
        "Dual (mean)": 21.9,
        "Dual (attention 0)": 9.9,
        "Dual-VideoMix (attention 0)": 7.7,
        "Dual (attention 1)": 22.3,
        "Dual (attention 2)": 22.2,
        "ViLt (mean)": 18.6,
        "Vita (mean)": 7.9,
        "STT (mean)": 14.2,
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
    print(r"\label{tab:hmdb51_zero-shot}")
    print(r"\end{table}")
