# HPC Submit Scripts Organization

This directory contains job submission scripts for running various tasks on the HPC cluster.

## Directory Structure

```
hpc_submit/
├── train_models/       # Scripts for training machine learning models
│   ├── dual_coop.sh
│   ├── video_coop.sh
│   ├── vidop.sh
│   ├── vilt.sh
│   ├── vita.sh
│   └── stt.sh
├── evaluation/         # Scripts for model evaluation tasks
│   ├── evaluate_model.sh
│   ├── mixture_model.sh
│   ├── run_multiple_evaluations.sh
│   └── run_multiple_augmentation_evaluations.sh
├── explainability/     # Scripts for model interpretation and analysis
│   ├── calibration.sh
│   └── explainer.sh
├── data/              # Scripts for data processing and downloading
│   ├── download_data.sh
│   ├── download_kinetics_train.sh
│   ├── extract_kinetics.sh
│   ├── process_data.sh
│   └── test_dataloader.sh
├── utilities/         # Utility and testing scripts
│   ├── cleanup_wandb.sh
│   ├── simple_inference.sh
│   └── time_test.sh
├── load_modules.sh    # Module loading script (kept at root)
└── logs/             # Job output logs

```

## Script-to-Notebook Mapping

All scripts now reference the corresponding organized notebook structure:

- `train_models/*.sh` → `scripts/train_models/*.py`
- `evaluation/*.sh` → `scripts/evaluation/*.py`
- `explainability/*.sh` → `scripts/explainability/*.py`
- `data/*.sh` → `scripts/data/*.py` (where applicable)

## Known Issues

The following scripts reference Python files that don't currently exist:
- `utilities/simple_inference.sh` → `scripts/video_inference.py` (missing)
- `utilities/time_test.sh` → `scripts/transformers_test.py` (missing)
- `data/extract_kinetics.sh` → `scripts/extracting_kinetics.py` (missing)

## Usage

To submit a job, navigate to the appropriate subdirectory and run:
```bash
bsub < script_name.sh
```

Or from the root directory:
```bash
bsub < hpc_submit/category/script_name.sh
```
