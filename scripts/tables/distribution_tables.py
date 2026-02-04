import math
import json
from collections import defaultdict

def analyze_models(models: dict, threshold=0.5, min_models=1, metric="accuracy"):
    below_counts = defaultdict(int)
    total_models = len(models)

    for name, path in models.items():
        if not path:
            continue

        with open(path, "r") as f:
            scores = json.load(f)

        for cls, metrics in scores.items():
            acc = metrics[metric]
            if acc < threshold:
                below_counts[cls] += 1

    classes_below = [
        cls for cls, count in below_counts.items()
        if count >= min_models
    ]

    return dict(below_counts), classes_below


if __name__ == "__main__":
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    metric = "accuracy"

    models = {
        # "Vidop Attention 0": "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/vidop/ucf101/best-no-hand-features-true-val/ucf101/None/101/all/False/False/test/metrics_per_class.json",
        # "CoOp Mean":        "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/video_coop/ucf101/2025-12-21_19-25-13/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        "Co a0": "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/video_coop/ucf101/2025-12-22_02-22-31/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        "Co a1": "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/video_coop/ucf101/2025-12-23_13-42-43/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        "Co a2": "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/video_coop/ucf101/2025-12-23_19-16-12/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        "Du M":   "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/dual_coop/ucf101/2025-12-22_00-47-07/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        "Du a0": "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/dual_coop/ucf101/2026-01-06_10-01-28/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        "Du a1": "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/dual_coop/ucf101/2026-01-05_17-04-16/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        "Du a2": "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/dual_coop/ucf101/2026-01-05_19-42-08/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        # "ViLT":             "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/vilt/ucf101/best-mean/ucf101/None/101/all/False/False/test/True/metrics_per_class.json",
        # "ViTa":             "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/vita/ucf101/2025-12-23_08-52-46/ucf101/None/None/all/False/False/test/metrics_per_class.json",
        # "STT":              "/zhome/de/d/169059/vlms-initial-testing/data/processed/results/stt/ucf101/2025-12-21_17-29-50/ucf101/None/None/all/False/False/test/metrics_per_class.json",
    }
    total_models = len(models)


    # ---------- TABLE 1: per-model counts ----------
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{" + "c" * (len(models) + 1) + "}")
    print(r"\hline")
    print(" & " + " & ".join(models.keys()) + r" \\")
    print(r"\hline")

    for threshold in thresholds:
        per_model = {}
        for name, path in models.items():
            with open(path, "r") as f:
                scores = json.load(f)
            per_model[name] = sum(
                1 for metrics in scores.values()
                if metrics[metric] < threshold
            )

        row = f"{int(threshold*100)}\\% & " + " & ".join(str(per_model[m]) for m in models)
        print(row + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Number of classes below each accuracy threshold per model. Co is CoOp, Du is Dual, shortened for space.}")
    print(r"\label{tab:classes_under_threshold_per_model}")
    print(r"\end{table}")
    print()


    # ---------- TABLE 2: overlapping classes ----------
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{" + "c" * (total_models + 1) + "}")
    print(r"\hline")
    print(" & " + " & ".join(str(k) for k in range(1, total_models + 1)) + r" \\")
    print(r"\hline")
    all_classes = set()
    for threshold in thresholds:
        values = []
        for k in range(1, total_models + 1):
            _, classes = analyze_models(
                models, threshold=threshold, min_models=k, metric=metric
            )
            values.append(str(len(classes)))
            all_classes.update(classes)

        row = f"{int(threshold*100)}\\% & " + " & ".join(values)
        print(row + r" \\")

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Number of overlapping classes below each accuracy threshold.}")
    print(r"\label{tab:overlap_classes_by_threshold}")
    print(r"\end{table}")

    print()
    print("All classes below thresholds:")
    for cls in sorted(all_classes):
        print(cls)
    

    print()
    print("Classes above thresholds for ALL models:")
    for threshold in thresholds[-1:]:
        above_all = None
        for name, path in models.items():
            with open(path, "r") as f:
                scores = json.load(f)

            current = {
                cls for cls, metrics in scores.items()
                if metrics[metric] >= threshold
            }

            above_all = current if above_all is None else above_all & current

        print(f"\n>= {int(threshold*100)}%:")
        for cls in sorted(above_all):
            print(cls)


