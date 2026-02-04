"""
Class weighting utilities for handling imbalanced datasets.
"""
from collections import defaultdict
from typing import Optional


def put_class_weights_in_config(
    config: dict,
    class_weighting: Optional[str],
    samples: list[tuple[str, str | int]],
    class_idxs: list[str]
):
    """Calculate and store class weights in configuration.
    
    Args:
        config: Configuration dictionary to update
        class_weighting: Weighting strategy ("balanced-per-class" or None)
        samples: List of (data, label) tuples
        class_idxs: List of class indices
    """
    if class_weighting is None:
        return
    
    class_counts = defaultdict(lambda: 0)
    for _, label in samples:
        class_counts[label] += 1

    if class_weighting == "balanced-per-class":
        total = sum(class_counts.values())
        num_classes = len(class_idxs)
        class_weights = [
            total / (num_classes * class_counts[cls]) if class_counts[cls] > 0 else 0
            for cls in class_idxs
        ]
        print(f"Using balanced-per-class class weights: {class_weights}")
        config["dataset-config"]["class-weights"] = class_weights
