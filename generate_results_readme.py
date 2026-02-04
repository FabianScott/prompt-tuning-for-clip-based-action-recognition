#!/usr/bin/env python3
"""
Generate a comprehensive README.md with all results, tables, and figures.
Run this script to update the RESULTS.md file with the latest data.

Usage:
    python generate_results_readme.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "tables"))

# Import results data directly
import results_data

FIGURES_DIR = PROJECT_ROOT / "figures"


def format_value(val: Optional[float]) -> str:
    """Format a numerical value, handling None."""
    if val is None:
        return "-"
    return f"{val:.1f}"


def create_markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Create a markdown table from headers and rows."""
    if not headers or not rows:
        return ""
    
    # Create header row
    header_row = "| " + " | ".join(headers) + " |"
    
    # Create separator
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    
    # Create data rows
    data_rows = []
    for row in rows:
        data_rows.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    return "\n".join([header_row, separator] + data_rows)


def generate_ucf101_main_results() -> str:
    """Generate UCF101 main results table."""
    headers = ["Model", "Train", "Val", "Test", "Hours"]
    rows = []
    
    for model, values in results_data.UCF101_MAIN_RESULTS.items():
        train, val, test, hours = values
        rows.append([
            model,
            format_value(train),
            format_value(val),
            format_value(test),
            str(hours) if hours else "-"
        ])
    
    return create_markdown_table(headers, rows)


def generate_ucf101_videomix_results() -> str:
    """Generate UCF101 VideoMix results table."""
    headers = ["Model", "Train", "Val", "Test", "Hours"]
    rows = []
    
    for model, values in results_data.UCF101_VIDEOMIX_RESULTS.items():
        train, val, test, hours = values
        rows.append([
            model,
            format_value(train),
            format_value(val),
            format_value(test),
            str(hours) if hours else "-"
        ])
    
    return create_markdown_table(headers, rows)


def generate_ucf101_1_temporal_view() -> str:
    """Generate UCF101 1 temporal view results table."""
    baseline = results_data.BASELINE_TEST
    headers = ["Model", "Test (1 temporal)", "Delta"]
    rows = []
    
    for model, values in results_data.UCF101_1_TEMPORAL_VIEW.items():
        test_acc = values[0]
        # Calculate delta if baseline exists
        delta = ""
        if model in baseline:
            delta_val = test_acc - baseline[model]
            delta = f"{delta_val:+.1f}"
        
        rows.append([
            model,
            format_value(test_acc),
            delta
        ])
    
    return create_markdown_table(headers, rows)


def generate_ucf101_combined_augmentation() -> str:
    """Generate UCF101 combined augmentation results table."""
    headers = ["Model", "Test (Combined Aug)"]
    rows = []
    
    for model, test_acc in results_data.UCF101_COMBINED_AUGMENTATION.items():
        rows.append([
            model,
            format_value(test_acc)
        ])
    
    return create_markdown_table(headers, rows)


def generate_ucf101_individual_augmentations() -> str:
    """Generate UCF101 individual augmentations table."""
    headers = ["Model", "H-Flip", "Colour Jitter", "Blur", "S&P", "Cutout p=0.1", "Cutout p=1"]
    rows = []
    
    for model, values in results_data.UCF101_INDIVIDUAL_AUGMENTATIONS.items():
        rows.append([model] + [format_value(v) for v in values])
    
    return create_markdown_table(headers, rows)


def generate_ucf101_train_augmented() -> str:
    """Generate UCF101 training with augmentations table."""
    headers = ["Model", "Train", "Val", "Test", "Hours"]
    rows = []
    
    for model, values in results_data.UCF101_TRAIN_AUGMENTED.items():
        train, val, test, hours = values
        rows.append([
            model,
            format_value(train),
            format_value(val),
            format_value(test),
            str(hours) if hours else "-"
        ])
    
    return create_markdown_table(headers, rows)


def generate_ucf101_removed_black_strips() -> str:
    """Generate UCF101 removed black strips results table."""
    headers = ["Model", "Test (No Black Strips)"]
    rows = []
    
    for model, values in results_data.UCF101_REMOVED_BLACK_STRIPS.items():
        rows.append([
            model,
            format_value(values[0])
        ])
    
    return create_markdown_table(headers, rows)


def generate_distribution_tables() -> str:
    """Generate both distribution tables."""
    output = []
    
    # Table 1: Classes below threshold per model
    data = results_data.CLASSES_UNDER_THRESHOLD
    headers = ["Threshold"] + list(data["models"].keys())
    rows = []
    
    for i, threshold in enumerate(data["thresholds"]):
        row = [f"{threshold}%"]
        for model in data["models"].keys():
            row.append(str(data["models"][model][i]))
        rows.append(row)
    
    output.append("**Number of classes below each accuracy threshold per model across UCF101 test set**\n")
    output.append(create_markdown_table(headers, rows))
    
    # Table 2: Overlapping difficult classes
    data2 = results_data.OVERLAP_CLASSES_BY_THRESHOLD
    headers2 = ["Threshold"] + [str(m) for m in data2["min_models"]]
    rows2 = []
    
    for i, threshold in enumerate(data2["thresholds"]):
        row = [f"{threshold}%"]
        for val in data2["overlaps"][i]:
            row.append(str(val))
        rows2.append(row)
    
    output.append("\n**Number of overlapping difficult classes by minimum model agreement**\n")
    output.append(create_markdown_table(headers2, rows2))
    
    return "\n".join(output)


def generate_kinetics400_zero_shot() -> str:
    """Generate Kinetics400 zero-shot results table."""
    headers = ["Model", "Top-1 Accuracy"]
    rows = []
    
    for model, acc in results_data.KINETICS400_ZERO_SHOT.items():
        rows.append([
            model,
            format_value(acc)
        ])
    
    return create_markdown_table(headers, rows)


def generate_kinetics400_kshot() -> str:
    """Generate Kinetics400 K-shot results table."""
    headers = ["Model", "16-shot Val", "16-shot Test", "4-shot Val", "4-shot Test"]
    rows = []
    
    for model, values in results_data.KINETICS400_KSHOT.items():
        rows.append([model] + [format_value(v) for v in values])
    
    return create_markdown_table(headers, rows)


def generate_hmdb51_zero_shot() -> str:
    """Generate HMDB51 zero-shot results table."""
    headers = ["Model", "Top-1 Accuracy"]
    rows = []
    
    for model, acc in results_data.HMDB51_ZERO_SHOT.items():
        rows.append([
            model,
            format_value(acc)
        ])
    
    return create_markdown_table(headers, rows)


def generate_hmdb51_kshot() -> str:
    """Generate HMDB51 K-shot results table."""
    headers = ["Model", "16-shot Val", "16-shot Test", "4-shot Val", "4-shot Test"]
    rows = []
    
    for model, values in results_data.HMDB51_KSHOT.items():
        rows.append([model] + [format_value(v) for v in values])
    
    return create_markdown_table(headers, rows)


def get_calibration_figures() -> List[Dict[str, str]]:
    """Get list of calibration figures."""
    figures = []
    cal_dir = FIGURES_DIR / "calibration" / "ucf101"
    
    if cal_dir.exists():
        for model_dir in sorted(cal_dir.iterdir()):
            if model_dir.is_dir():
                # Find all timestamp subdirectories and use the latest one
                timestamp_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
                if timestamp_dirs:
                    latest_dir = max(timestamp_dirs, key=lambda d: d.name)
                    # Get PNG files from the latest timestamp directory
                    for fig in sorted(latest_dir.glob("*.png")):
                        figures.append({
                            "path": str(fig.relative_to(PROJECT_ROOT)),
                            "name": f"{model_dir.name} - {fig.stem}",
                            "model": model_dir.name
                        })
    
    return figures


def get_gflops_figures() -> List[Dict[str, str]]:
    """Get list of GFLOPs figures."""
    figures = []
    gflops_dir = FIGURES_DIR / "gflops"
    
    if gflops_dir.exists():
        for fig in sorted(gflops_dir.glob("*.png")):
            figures.append({
                "path": str(fig.relative_to(PROJECT_ROOT)),
                "name": fig.stem.replace('_', ' ').title()
            })
    
    return figures


def get_explainer_figures() -> Dict[str, List[str]]:
    """Get list of explainer figures organized by model."""
    figures = {}
    explainer_dir = FIGURES_DIR / "explainer"
    
    if explainer_dir.exists():
        for model_dir in sorted(explainer_dir.iterdir()):
            if model_dir.is_dir():
                model_name = model_dir.name
                figures[model_name] = []
                
                # Get a sample of figures (first few from each action)
                actions = {}
                for fig in sorted(model_dir.rglob("*.png")):
                    action = fig.parent.name
                    if action not in actions:
                        actions[action] = []
                    actions[action].append(str(fig.relative_to(PROJECT_ROOT)))
                
                # Take first 2 examples from first 3 actions
                for action, paths in sorted(actions.items())[:3]:
                    figures[model_name].extend(paths[:2])
    
    return figures


def generate_readme() -> str:
    """Generate the complete README content."""
    
    readme = """# Prompt Tuning for CLIP-Based Action Recognition - Results

**Last Updated:** {date}

This document presents comprehensive results from experiments on video action recognition using prompt tuning techniques with CLIP-based models.

## Table of Contents

1. [UCF101 Results](#ucf101-results)
2. [Kinetics400 Results](#kinetics400-results)
3. [HMDB51 Results](#hmdb51-results)
4. [Computational Cost Analysis](#computational-cost-analysis)
5. [Explainability Analysis](#explainability-analysis)
6. [Calibration Results](#calibration-results)

---

## UCF101 Results

### Main Training Results

The following table shows train, validation and test accuracies on UCF101 for all model variants.

{ucf101_main}

### VideoMix Augmentation Results

Results for models trained using the VideoMix data augmentation technique:

{ucf101_videomix}

### Temporal View Analysis

Impact of using single vs. multiple temporal views:

{ucf101_temporal}

### Augmentation Robustness

#### Combined Augmentations

{ucf101_augmentation}

#### Individual Augmentation Effects

{ucf101_aug_separate}

#### Training with Augmentations

{ucf101_aug_train}

### Class-Level Performance Analysis

#### Classes Below Threshold Per Model

{distribution_per_model}

#### Overlapping Difficult Classes

{distribution_overlap}

### Black Strip Analysis

Impact of removing black strips from videos:

{black_strips}

---

## Kinetics400 Results

### Zero-Shot Transfer

Transfer learning from UCF101 to Kinetics400 without fine-tuning:

{kinetics_zero}

### K-Shot Learning

Few-shot learning on Kinetics400 (4-shot and 16-shot):

{kinetics_kshot}

---

## HMDB51 Results

### Zero-Shot Transfer

Transfer learning from UCF101 to HMDB51 without fine-tuning:

{hmdb_zero}

### K-Shot Learning

Few-shot learning on HMDB51 (4-shot and 16-shot):

{hmdb_kshot}

---

## Computational Cost Analysis

### GFLOPs vs. Accuracy

The following figures show the relationship between computational cost (GFLOPs) and model accuracy.

"""
    
    # Add GFLOPs figures
    gflops_figs = get_gflops_figures()
    for fig in gflops_figs:
        if 'vs_val_accuracy_101' in fig['path']:
            readme += f"\n#### {fig['name']}\n\n"
            readme += f"![{fig['name']}]({fig['path']})\n\n"
    
    for fig in gflops_figs:
        if 'vs_num_classes' in fig['path']:
            readme += f"\n#### {fig['name']}\n\n"
            readme += f"![{fig['name']}]({fig['path']})\n\n"
    
    readme += """
### Key Findings

- VidOp has the lowest computational cost due to no text context
- CoOp, Dual, ViLT, and Vita have similar GFLOPs growth rates
- STT grows 10x faster than other models with increasing number of classes
- At 400 classes, STT uses double the GFLOPs of simpler models

---

## Explainability Analysis

### Attention Rollout Visualizations

Attention rollout reveals how models attend to different parts of video frames during classification.
It multiplies attention weights across layers to highlight important regions.
Here Four versions are used:
- Attention Rollout
- Attention Rollout wrt to the CLS token
- Attention Rollout weighted by the CLS token
- GradCAM (no meaningful flow found)

"""
    
    # Add explainer figures (sample)
    explainer_figs = get_explainer_figures()
    sample_models = ['dual_coopbest-attention-1-ucf101-1-attention-rollout-cls',
                     'video_coopbest-attention-0-ucf101-1-attention-rollout-cls',
                     'sttbest-true-val-ucf101-1-attention-rollout-cls']
    
    for model in sample_models:
        if model in explainer_figs:
            readme += f"\n#### {model}\n\n"
            for fig_path in explainer_figs[model][:4]:  # Show first 4
                action = Path(fig_path).parent.name
                readme += f"**{action}**\n\n"
                readme += f"![{action}]({fig_path})\n\n"
    
    readme += f"""
### Key Findings

- Models attend to black strips when present in videos
- Attention patterns don't always align with human-interpretable features
- Removing black strips significantly impacts some models (Dual mean: -11%, Dual a-2: -18.6%)
- GradCAM returns all zeros due to gradient flow through classification token

---

## Calibration Results

Model calibration analysis showing confidence vs. accuracy:

"""
    
    # Add calibration figures in a grid (3 columns)
    cal_figs = get_calibration_figures()
    if cal_figs:
        readme += "<table>\n"
        for i in range(0, len(cal_figs), 3):
            readme += "<tr>\n"
            for fig in cal_figs[i:i+3]:
                readme += f"<td align='center'><b>{fig['model']}</b><br/><img src='{fig['path']}' width='300'/></td>\n"
            readme += "</tr>\n"
        readme += "</table>\n\n"
    
    readme += """
---

## Summary of Key Findings

### Best Performing Models

1. **Dual (attention-1)**: 93.2% test accuracy on UCF101
2. **CoOp (attention-1)**: 92.5% test accuracy on UCF101
3. **STT (mean)**: 92.0% test accuracy on UCF101

### Model Groups by Performance

- **Group 1 (<65%)**: VidOp with mean pooling or handcrafted features
- **Group 2 (80-90%)**: VidOp a-0, CoOp mean, ViLT
- **Group 3 (90-95%)**: CoOp/Dual with attention, Vita, STT

### Robustness Findings

- Models are highly sensitive to color jitter (40-60% accuracy drop)
- Salt and pepper noise causes 16-40% drop for most models
- Horizontal flip and cutout have minimal impact (1-3%)
- VideoMix augmentation increases robustness for ViLT, Vita, and STT

### Transfer Learning

- Dual models transfer best to Kinetics400 K-shot setting
- Zero-shot transfer from UCF101 shows limited success (<25% on HMDB51)
- K-shot learning significantly improves transfer performance

---

## Reproducing Results

All tables in this document are generated from scripts in `notebooks/tables/`:

```bash
# Generate all tables
python notebooks/tables/generate_all_tables.py

# Verify all tables are covered
python notebooks/tables/verify_coverage.py

# Generate individual table
python notebooks/tables/ucf101_main_results.py
```

See [notebooks/tables/README.md](notebooks/tables/README.md) for complete documentation.

---

**Note**: This README is automatically generated. To update with latest results:

```bash
python generate_results_readme.py
```
"""
    
    return readme


def main():
    """Generate and save the README."""
    from datetime import datetime
    
    print("Generating Results README...")
    print("=" * 80)
    
    # Generate table content using direct imports
    print("Generating tables from data dictionaries...")
    
    dist_tables = generate_distribution_tables()
    dist_parts = dist_tables.split('\n\n**Number of overlapping')
    
    tables = {
        'ucf101_main': generate_ucf101_main_results(),
        'ucf101_videomix': generate_ucf101_videomix_results(),
        'ucf101_temporal': generate_ucf101_1_temporal_view(),
        'ucf101_augmentation': generate_ucf101_combined_augmentation(),
        'ucf101_aug_separate': generate_ucf101_individual_augmentations(),
        'ucf101_aug_train': generate_ucf101_train_augmented(),
        'distribution_per_model': dist_parts[0] if len(dist_parts) > 0 else "",
        'distribution_overlap': '**Number of overlapping' + dist_parts[1] if len(dist_parts) > 1 else "",
        'black_strips': generate_ucf101_removed_black_strips(),
        'kinetics_zero': generate_kinetics400_zero_shot(),
        'kinetics_kshot': generate_kinetics400_kshot(),
        'hmdb_zero': generate_hmdb51_zero_shot(),
        'hmdb_kshot': generate_hmdb51_kshot(),
    }
    
    # Generate README content
    print("Generating README content...")
    readme = generate_readme()
    
    # Format with current date
    readme = readme.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **tables
    )
    
    # Save to file
    output_path = PROJECT_ROOT / "RESULTS.md"
    with open(output_path, 'w') as f:
        f.write(readme)
    
    print(f"\n✓ README generated successfully: {output_path}")
    print(f"  Size: {len(readme):,} characters")
    print("=" * 80)


if __name__ == "__main__":
    main()
