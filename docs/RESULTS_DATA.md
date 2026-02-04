# Results Data Structure Documentation

## Overview

All experimental results are now stored as Python dictionaries in [src/tables/results_data.py](../src/tables/results_data.py), making them easily accessible for analysis, visualization, and report generation without needing to parse LaTeX or run subprocess scripts.

## Benefits of This Approach

1. **Direct Access**: Import and use results as Python dictionaries
2. **No LaTeX Parsing**: Eliminates complex and error-prone LaTeX parsing logic
3. **Type Safety**: Clear data structures with documented formats
4. **Fast**: No subprocess overhead or script execution needed
5. **Flexible**: Easy to create custom analyses, visualizations, and reports
6. **Single Source of Truth**: All data in one centralized location

## Available Datasets

### UCF101 Results

#### `UCF101_MAIN_RESULTS`
Main training results on UCF101.

**Format**: `Dict[str, List[float]]` where each value is `[train_acc, val_acc, test_acc, train_hours]`

**Example**:
```python
from src.tables import results_data

# Get results for a specific model
train, val, test, hours = results_data.UCF101_MAIN_RESULTS["CoOp (attention 1)"]
print(f"Test accuracy: {test}%")
```

#### `UCF101_VIDEOMIX_RESULTS`
Results with VideoMix augmentation.

**Format**: `Dict[str, List[Optional[float]]]` where each value is `[train_acc, val_acc, test_acc, train_hours]`

#### `UCF101_1_TEMPORAL_VIEW`
Results using single temporal view.

**Format**: `Dict[str, List[float]]` where each value is `[test_acc]`

#### `BASELINE_TEST`
Baseline test accuracies for delta calculations.

**Format**: `Dict[str, float]` mapping model name to test accuracy

**Example**:
```python
# Calculate improvement from baseline
model = "Dual (attention 1)"
temporal_acc = results_data.UCF101_1_TEMPORAL_VIEW[model][0]
baseline = results_data.BASELINE_TEST[model]
delta = temporal_acc - baseline
print(f"Delta: {delta:+.1f}%")
```

#### `UCF101_COMBINED_AUGMENTATION`
Test accuracy under combined augmentations.

**Format**: `Dict[str, float]` mapping model name to test accuracy

#### `UCF101_INDIVIDUAL_AUGMENTATIONS`
Performance under individual augmentation types.

**Format**: `Dict[str, List[float]]` where each value is:
`[H-Flip, Colour Jitter, Blur, S&P, Cutout p=0.1, Cutout p=1]`

**Example**:
```python
# Compare color jitter robustness
for model, augs in results_data.UCF101_INDIVIDUAL_AUGMENTATIONS.items():
    color_jitter_acc = augs[1]  # Index 1 is color jitter
    print(f"{model}: {color_jitter_acc}%")
```

#### `UCF101_TRAIN_AUGMENTED`
Results when training with augmentations.

**Format**: `Dict[str, List[float]]` where each value is `[train_acc, val_acc, test_acc, train_hours]`

#### `UCF101_REMOVED_BLACK_STRIPS`
Test accuracy with black strips removed from videos.

**Format**: `Dict[str, List[float]]` where each value is `[test_acc]`

### Distribution Analysis

#### `CLASSES_UNDER_THRESHOLD`
Number of classes below each accuracy threshold per model.

**Format**:
```python
{
    "thresholds": [50, 60, 70, 80, 85, 90, 95],
    "models": {
        "Co a0": [2, 5, 8, 13, 21, 27, 41],
        # ... more models
    }
}
```

**Example**:
```python
# Classes below 90% for each model
data = results_data.CLASSES_UNDER_THRESHOLD
threshold_90_idx = data["thresholds"].index(90)

for model, counts in data["models"].items():
    print(f"{model}: {counts[threshold_90_idx]} classes below 90%")
```

#### `OVERLAP_CLASSES_BY_THRESHOLD`
Number of overlapping difficult classes by minimum model agreement.

**Format**:
```python
{
    "thresholds": [50, 60, 70, 80, 85, 90, 95],
    "min_models": [1, 2, 3, 4, 5, 6, 7],
    "overlaps": [
        [5, 3, 2, 0, 0, 0, 0],  # For 50% threshold
        # ... more thresholds
    ]
}
```

### Kinetics400 Results

#### `KINETICS400_ZERO_SHOT`
Zero-shot transfer results.

**Format**: `Dict[str, float]` mapping model name to top-1 accuracy

#### `KINETICS400_KSHOT`
K-shot learning results.

**Format**: `Dict[str, List[Optional[float]]]` where each value is:
`[16-shot-val, 16-shot-test, 4-shot-val, 4-shot-test]`

### HMDB51 Results

#### `HMDB51_ZERO_SHOT`
Zero-shot transfer results.

**Format**: `Dict[str, float]` mapping model name to top-1 accuracy

#### `HMDB51_KSHOT`
K-shot learning results.

**Format**: `Dict[str, List[Optional[float]]]` where each value is:
`[16-shot-val, 16-shot-test, 4-shot-val, 4-shot-test]`

## Usage Examples

### Basic Access

```python
import sys
from pathlib import Path

# Add tables directory to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "tables"))
import results_data

# Get specific result
train, val, test, hours = results_data.UCF101_MAIN_RESULTS["Dual (attention 1)"]
print(f"Best model: {test}% in {hours} hours")
```

### Finding Best Model

```python
best_model = max(
    results_data.UCF101_MAIN_RESULTS.items(),
    key=lambda x: x[1][2]  # x[1][2] is test accuracy
)
print(f"Best: {best_model[0]} with {best_model[1][2]}%")
```

### Comparing Augmentation Robustness

```python
# Compare S&P noise robustness (index 3)
sp_results = {
    model: values[3]
    for model, values in results_data.UCF101_INDIVIDUAL_AUGMENTATIONS.items()
}

for model, acc in sorted(sp_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model}: {acc}%")
```

### Custom Analysis

```python
# Calculate training efficiency (accuracy per hour)
efficiency = {}
for model, (train, val, test, hours) in results_data.UCF101_MAIN_RESULTS.items():
    if hours and test:
        efficiency[model] = test / hours

top_3 = sorted(efficiency.items(), key=lambda x: x[1], reverse=True)[:3]
for model, eff in top_3:
    print(f"{model}: {eff:.1f} acc/hour")
```

### Generating Tables

```python
def create_markdown_table(data_dict, value_index=0):
    """Generate markdown table from results dictionary."""
    rows = []
    for model, values in data_dict.items():
        value = values[value_index] if isinstance(values, list) else values
        rows.append(f"| {model} | {value:.1f} |")
    
    return "| Model | Accuracy |\n| --- | --- |\n" + "\n".join(rows)

# Generate table for zero-shot Kinetics400
table = create_markdown_table(results_data.KINETICS400_ZERO_SHOT)
print(table)
```

## Generating RESULTS.md

The [generate_results_readme.py](../generate_results_readme.py) script uses these dictionaries to generate the comprehensive RESULTS.md document:

```bash
python generate_results_readme.py
```

This script:
1. Imports all data from `results_data.py`
2. Generates markdown tables directly from dictionaries
3. Discovers and embeds figures from `figures/` directory
4. Creates a comprehensive results document

## Updating Results Data

To add new results or update existing ones:

1. Edit [src/tables/results_data.py](../src/tables/results_data.py)
2. Add your new dictionary or update existing values
3. Run `python generate_results_readme.py` to regenerate RESULTS.md

**Example**:
```python
# Add new experiment results
UCF101_NEW_EXPERIMENT = {
    "Model A": [95.0, 93.0, 92.0, 10],
    "Model B": [96.0, 94.0, 93.0, 12],
}
```

## Complete Example

See [example_usage.py](../example_usage.py) for a comprehensive demonstration of all access patterns and analysis techniques.

Run it with:
```bash
python example_usage.py
```

## Migration from Old System

The previous system required:
- Running subprocess scripts
- Parsing LaTeX output
- Complex regex patterns
- Multiple failure points

The new system:
- Direct dictionary imports
- No parsing needed
- Type-safe access
- Single source of truth

All table generation scripts in `notebooks/tables/` are now optional - they're kept for backwards compatibility and manual LaTeX generation if needed, but the primary workflow uses `results_data.py` directly.
