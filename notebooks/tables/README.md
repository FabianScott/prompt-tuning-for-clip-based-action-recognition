# Table Generation Scripts

This directory contains scripts to generate all LaTeX tables used in the Results chapter.

## Scripts and Corresponding Tables

### UCF101 Results

1. **`ucf101_main_results.py`** → `tab:ucf101_results`
   - Main training results with train/val/test accuracies and training times
   - All model variants including VidOp, CoOp, Dual, ViLt, Vita, and STT

2. **`ucf101_videomix_results.py`** → `tab:ucf101_results_videomix`
   - Results for models trained with VideoMix data augmentation
   - Includes Dual-VideoMix, ViLt-VideoMix, Vita-VideoMix, STT-VideoMix

3. **`1_temporal_view.py`** → `tab:ucf101_1_temporal_view`
   - Test accuracies using single temporal view instead of multiple views
   - Shows the impact of using fewer temporal views on performance

4. **`ucf101_combined_augmentation.py`** → `tab:ucf101_augmentation`
   - Test accuracies with combined augmentations (cutout, salt & pepper, horizontal flip)
   - Includes delta from non-augmented baseline

5. **`train_augmented.py`** → `tab:ucf101_results_augmented_train`
   - Training results when augmentations are applied to training data
   - Tests on non-augmented validation/test sets

6. **`individual_augmentation.py`** → `tab:ucf101_augmentations_separate_1` and `tab:ucf101_augmentations_separate_2`
   - Individual augmentation effects (H-Flip, Colour Jitter, Blur, S&P, Cutout)
   - Split into two tables for readability

7. **`removed_black_strip.py`** → `tab:ucf101_removed_black_strips`
   - Test accuracies when black strips are removed from videos
   - Shows model dependence on these artifacts

### Distribution Analysis

8. **`distribution_tables.py`** → `tab:classes_under_threshold_per_model` and `tab:overlap_classes_by_threshold`
   - Per-class accuracy analysis
   - Shows which models struggle with the same classes

### Kinetics400 Results

9. **`kinetics400_zero_shot.py`** → `tab:kinetics400_zero-shot`
   - Zero-shot transfer from UCF101 to Kinetics400

10. **`kinetics400_kshot.py`** → `tab:kinetics_kshot`
    - K-shot learning results (4-shot and 16-shot) on Kinetics400

### HMDB51 Results

11. **`hmdb51_zero_shot.py`** → `tab:hmdb51_zero-shot`
    - Zero-shot transfer from UCF101 to HMDB51

12. **`hmdb51_kshot.py`** → `tab:hmdb51_kshot`
    - K-shot learning results (4-shot and 16-shot) on HMDB51

## Usage

Run any script from the notebooks directory:

```bash
cd /zhome/de/d/169059/prompt-tuning-for-clip-based-action-recognition/notebooks
python tables/<script_name>.py
```

Or run from the project root:

```bash
python notebooks/tables/<script_name>.py
```

All scripts automatically handle path adjustments and will output LaTeX table code to stdout.

## Dependencies

Most scripts use the utility function in `src/tables/augmentation_utils.py` which provides:
- Baseline test/validation accuracies for computing deltas
- Helper function `print_augmented_results()` for formatting

## Utility Scripts

- **`verify_coverage.py`** - Verifies that all LaTeX table labels have corresponding generation scripts
- **`generate_all_tables.py`** - Master script to generate all tables at once with section headers

### Generate All Tables

To generate all tables at once:

```bash
cd /zhome/de/d/169059/prompt-tuning-for-clip-based-action-recognition/notebooks/tables
python generate_all_tables.py > ../../all_tables.tex
```

### Verify Coverage

To verify all tables are covered:

```bash
cd /zhome/de/d/169059/prompt-tuning-for-clip-based-action-recognition/notebooks/tables
python verify_coverage.py
```

## Table Coverage

All tables in the Results chapter LaTeX document now have corresponding generation scripts:
- ✅ tab:ucf101_results
- ✅ tab:ucf101_results_videomix
- ✅ tab:ucf101_1_temporal_view
- ✅ tab:classes_under_threshold_per_model
- ✅ tab:overlap_classes_by_threshold
- ✅ tab:ucf101_augmentation
- ✅ tab:ucf101_results_augmented_train
- ✅ tab:ucf101_augmentations_separate_1
- ✅ tab:ucf101_augmentations_separate_2
- ✅ tab:kinetics400_zero-shot
- ✅ tab:kinetics_kshot
- ✅ tab:hmdb51_zero-shot
- ✅ tab:hmdb51_kshot
- ✅ tab:ucf101_removed_black_strips
