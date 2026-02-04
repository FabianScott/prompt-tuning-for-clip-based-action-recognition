#!/usr/bin/env python3
"""
Example of how to import and use the results data directly.

This demonstrates how to access experimental results as Python dictionaries
without parsing LaTeX or running subprocess scripts.
"""

import sys
from pathlib import Path

# Add the tables directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src" / "tables"))

import results_data

def main():
    print("=" * 80)
    print("Example: Accessing Results Data as Python Dictionaries")
    print("=" * 80)
    
    # Example 1: Get UCF101 main results
    print("\n1. UCF101 Main Results for CoOp (attention 1):")
    model = "CoOp (attention 1)"
    train, val, test, hours = results_data.UCF101_MAIN_RESULTS[model]
    print(f"   Train: {train}%, Val: {val}%, Test: {test}%, Hours: {hours}")
    
    # Example 2: Find best performing model
    print("\n2. Best performing model on UCF101:")
    best_model = max(
        results_data.UCF101_MAIN_RESULTS.items(),
        key=lambda x: x[1][2]  # x[1][2] is test accuracy
    )
    print(f"   {best_model[0]}: {best_model[1][2]}% test accuracy")
    
    # Example 3: Compare models on specific augmentation
    print("\n3. Color Jitter robustness comparison:")
    # Color Jitter is index 1 in individual augmentations
    color_jitter_results = {}
    for model, values in results_data.UCF101_INDIVIDUAL_AUGMENTATIONS.items():
        color_jitter_results[model] = values[1]
    
    # Show top 3 most robust
    top_3 = sorted(color_jitter_results.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (model, acc) in enumerate(top_3, 1):
        print(f"   {i}. {model}: {acc}%")
    
    # Example 4: Zero-shot transfer performance
    print("\n4. Zero-shot transfer to HMDB51:")
    hmdb_results = results_data.HMDB51_ZERO_SHOT
    dual_models = {k: v for k, v in hmdb_results.items() if "Dual" in k}
    for model, acc in sorted(dual_models.items(), key=lambda x: x[1], reverse=True):
        print(f"   {model}: {acc}%")
    
    # Example 5: Distribution analysis
    print("\n5. Classes below 90% threshold:")
    for model, counts in results_data.CLASSES_UNDER_THRESHOLD["models"].items():
        # 90% threshold is at index 5
        count_90 = counts[5]
        print(f"   {model}: {count_90} classes")
    
    # Example 6: Custom analysis - training efficiency
    print("\n6. Test accuracy per training hour (UCF101):")
    efficiency = {}
    for model, values in results_data.UCF101_MAIN_RESULTS.items():
        train, val, test, hours = values
        if hours and test:
            efficiency[model] = test / hours
    
    top_efficient = sorted(efficiency.items(), key=lambda x: x[1], reverse=True)[:3]
    for i, (model, eff) in enumerate(top_efficient, 1):
        test_acc = results_data.UCF101_MAIN_RESULTS[model][2]
        hours = results_data.UCF101_MAIN_RESULTS[model][3]
        print(f"   {i}. {model}: {eff:.1f} acc/hour ({test_acc}% in {hours}h)")
    
    # Example 7: Access baseline for delta calculations
    print("\n7. Computing delta from baseline for 1 temporal view:")
    model = "Dual (attention 1)"
    test_1_temp = results_data.UCF101_1_TEMPORAL_VIEW[model][0]
    baseline = results_data.BASELINE_TEST[model]
    delta = test_1_temp - baseline
    print(f"   {model}:")
    print(f"   - Baseline (3 views): {baseline}%")
    print(f"   - 1 temporal view: {test_1_temp}%")
    print(f"   - Delta: {delta:+.1f}%")
    
    print("\n" + "=" * 80)
    print("See src/tables/results_data.py for all available datasets")
    print("=" * 80)


if __name__ == "__main__":
    main()
