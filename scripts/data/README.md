# Data Utilities

This folder contains scripts for downloading, processing, and testing datasets used in the project.

## Scripts

### `download_kinetics400.py`
Downloads the Kinetics-400 dataset using FiftyOne's zoo dataset functionality.

**Usage:**
```bash
python scripts/data/download_kinetics400.py
```

**Description:**
- Automatically downloads the Kinetics-400 training split
- Saves data to the configured raw data path
- Uses 8 workers for parallel downloading
- Includes retry logic for failed downloads

**Requirements:**
- `fiftyone` library installed
- Sufficient disk space for the dataset ~370 Gb
- Configured `DATA_RAW_PATH` in project settings

---

### `process_nwup.py`
Processes the NWPU (Northwestern Polytechnical University) dataset for action detection tasks.

**Usage:**
```bash
python scripts/data/process_nwup.py
```

**Description:**
- Generates action detection annotations from raw data
- Creates PyTorch-compatible dataset folder structure
- Initializes and validates the NWPUDataset
- Reports dataset statistics (number of samples and classes)

**Output:**
- Processed dataset ready for training
- Generated annotation files

---

### `test_dataloader.py`
Tests and validates data loaders for various datasets and methods.

**Usage:**
```python
from scripts.data.test_dataloader import test_dataloader

test_dataloader(
    method_default="video_coop",
    dataset_default="ucf101",
    train_first=True,
    fixed_config=None,
    break_after=10,  # Optional: limit batches
    idx_to_plot=0    # Optional: plot specific sample
)
```

**Description:**
- Tests train, validation, and test data loaders
- Validates batch shapes and data integrity
- Can optionally plot sample frames from batches
- Useful for debugging data pipeline issues

**Parameters:**
- `method_default`: Model method (e.g., "video_coop", "dual_coop", "stt")
- `dataset_default`: Dataset name (e.g., "ucf101", "hmdb51", "kinetics400")
- `train_first`: Whether to test training loader first
- `fixed_config`: Optional dictionary to override default configurations
- `break_after`: Optional limit on number of batches to test
- `idx_to_plot`: Index of sample to visualize

## Notes

- All scripts automatically adjust the working directory to the project root
- Requires project dependencies installed (see `requirements.txt`)
- Data paths are configured in `src/configs/paths.py`
