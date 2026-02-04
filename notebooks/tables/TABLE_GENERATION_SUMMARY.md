# Table Generation Summary

## Overview

All LaTeX tables from the Results chapter have corresponding Python generation scripts in the `notebooks/tables/` directory.

## Complete Table-to-Script Mapping

| LaTeX Label | Script | Description |
|------------|--------|-------------|
| `tab:ucf101_results` | `ucf101_main_results.py` | Main UCF101 training results |
| `tab:ucf101_results_videomix` | `ucf101_videomix_results.py` | VideoMix augmentation results |
| `tab:ucf101_1_temporal_view` | `1_temporal_view.py` | Single temporal view results |
| `tab:ucf101_augmentation` | `ucf101_combined_augmentation.py` | Combined augmentations on test set |
| `tab:ucf101_results_augmented_train` | `train_augmented.py` | Training with augmentations |
| `tab:ucf101_augmentations_separate_1` | `individual_augmentation.py` | Individual augmentations (part 1) |
| `tab:ucf101_augmentations_separate_2` | `individual_augmentation.py` | Individual augmentations (part 2) |
| `tab:ucf101_removed_black_strips` | `removed_black_strip.py` | Results without black strips |
| `tab:classes_under_threshold_per_model` | `distribution_tables.py` | Per-model class accuracy distribution |
| `tab:overlap_classes_by_threshold` | `distribution_tables.py` | Overlapping difficult classes |
| `tab:kinetics400_zero-shot` | `kinetics400_zero_shot.py` | Kinetics400 zero-shot transfer |
| `tab:kinetics_kshot` | `kinetics400_kshot.py` | Kinetics400 K-shot learning |
| `tab:hmdb51_zero-shot` | `hmdb51_zero_shot.py` | HMDB51 zero-shot transfer |
| `tab:hmdb51_kshot` | `hmdb51_kshot.py` | HMDB51 K-shot learning |

## Quick Start

### Generate a Single Table

```bash
cd notebooks/tables
python ucf101_main_results.py
```

### Generate All Tables

```bash
cd notebooks/tables
python generate_all_tables.py > ../../all_tables.tex
```

### Verify Coverage

```bash
cd notebooks/tables
python verify_coverage.py
```

## Files Added

### Generation Scripts
- `ucf101_main_results.py` - NEW
- `ucf101_videomix_results.py` - NEW
- `ucf101_combined_augmentation.py` - NEW
- `kinetics400_zero_shot.py` - NEW
- `kinetics400_kshot.py` - NEW
- `hmdb51_zero_shot.py` - NEW
- `hmdb51_kshot.py` - NEW

### Pre-existing Scripts
- `1_temporal_view.py` - Already existed
- `distribution_tables.py` - Already existed
- `individual_augmentation.py` - Already existed
- `removed_black_strip.py` - Already existed
- `train_augmented.py` - Already existed

### Utility Scripts
- `verify_coverage.py` - NEW - Verification tool
- `generate_all_tables.py` - NEW - Master generation script
- `README.md` - NEW - Documentation

## Status

✅ **All 14 tables are covered by generation scripts**

Every table referenced in the Results chapter LaTeX document now has a corresponding Python script that generates it programmatically.
