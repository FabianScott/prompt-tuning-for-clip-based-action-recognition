# Baseline (un-augmented test accuracies) from Table 2
baseline_test = {
    "VidOp (attention 0)": 82.9,
    "CoOp (mean)": 85.6,
    "CoOp (attention 0)": 91.9,
    "CoOp (attention 1)": 92.5,
    "CoOp (attention 2)": 91.7,
    "Dual (mean)": 91.7,
    "Dual (attention 0)": 91.3,
    "Dual (attention 1)": 93.2,
    "Dual (attention 2)": 90.7,
    "ViLt": 84.8,
    "ViTa": 90.8,
    "STT": 92.0,
    "Dual-VideoMix (attention 0)": 91.3,
    "ViLt-VideoMix": 85.1,
    "ViTa-VideoMix": 86.8,
    "STT-VideoMix": 91.5,
}

baseline_val = {
    "VidOp (mean)": 54.4,
    "VidOp H (mean)": 52.5,
    "VidOp (attention 0)": 84.3,
    "VidOp H (attention 0)": 58.0,
    "CoOp (mean)": 86.8,
    "CoOp (attention 0)": 93.3,
    "CoOp (attention 1)": 93.5,
    "CoOp (attention 2)": 92.6,
    "Dual (mean)": 93.3,
    "Dual (attention 0)": 93.2,
    "Dual (attention 1)": 95.0,
    "Dual (attention 2)": 95.0,
    "ViLt": 86.5,
    "ViTa": 91.0,
    "STT": 94.9,
    "Dual-VideoMix (attention 0)": 92.3,
    "ViLt-VideoMix": 87.0,
    "ViTa-VideoMix": 90.8,
    "STT-VideoMix": 91.5,
}


baselines = {
    "val": baseline_val,
    "test": baseline_test,
}


def print_augmented_results(augmented: dict, kinds: list):
    for model, values in augmented.items():
        formatted = []
        for v, kind in zip(values, kinds):
            if kind not in baselines:
                formatted.append(f"{v:.1f}")
                continue
            base = baselines[kind][model]
            diff = v - base
            formatted.append(f"{{{v:.1f} ({diff:+.1f})}}")
        print(
            model.replace("VideoMix", "V").replace("attention ", "a-"),
            " & ",
            " & ".join(formatted),
            r" \\"
        )

