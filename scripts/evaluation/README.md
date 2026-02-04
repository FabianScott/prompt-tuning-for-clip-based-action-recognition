# Model Evaluation

This folder contains scripts for evaluating trained models on various datasets and analyzing their performance.

## Scripts

### `evaluate_model.py`
Evaluates a single trained model on a specified dataset.

**Usage:**
```bash
python scripts/evaluation/evaluate_model.py
```

**Configuration:**
Edit the `keys_list` in the script with model keys in the format:
```
"method_name/dataset_train/run_id"
```

Example keys:
- `"video_coop/ucf101/best-attention-0"`
- `"dual_coop/kinetics400/k-4-attention-1"`
- `"stt/ucf101/best-true-val"`

**Evaluation Config Options:**
- `model_name`: Model method (video_coop, dual_coop, stt, vilt, vita, vidop)
- `dataset_train`: Dataset used for training
- `run_id`: Specific run identifier
- `dataset_test`: Dataset to evaluate on
- `K-test`: Number of samples per class (None for full dataset)
- `num-classes`: Number of classes to use (None for all)
- `use-augmentation`: Whether to use test-time augmentation
- `split`: Dataset split to evaluate ("test", "val")

**Meta Config Options:**
- `batch-size`: Batch size for evaluation
- `num-workers`: Number of data loading workers

---

### `run_multiple_evaluations.py`
Batch evaluation script for testing multiple models across different settings.

**Usage:**
```bash
python scripts/evaluation/run_multiple_evaluations.py
```

**Description:**
- Evaluates multiple models sequentially
- Can vary evaluation settings (dataset, augmentation, etc.)
- Includes error handling and progress tracking
- Logs results to WandB (if enabled)

**Configuration:**
- Modify `keys_list` to include all models to evaluate
- Adjust `settings_to_vary` to test different configurations
- Each model is evaluated with all setting combinations

---

### `run_multiple_augmentation_evaluations.py`
Specialized script for evaluating models with different augmentation strategies.

**Usage:**
```bash
python scripts/evaluation/run_multiple_augmentation_evaluations.py
```

**Description:**
- Tests models under various augmentation conditions
- Useful for understanding model robustness
- Compares performance with and without augmentation

---

### `flop_analysis.py`
Analyzes computational complexity (FLOPs) of trained models.

**Usage:**
```python
from scripts.evaluation.flop_analysis import flop_analysis

flop_analysis(
    model_keys=["video_coop/ucf101/best-attention-0"],
    name_for_display="Video-CoOp",
    dataset_name="ucf101",
    plot_savename="gflops_analysis.png",
    num_classes=None,
    device=torch.device("cpu")
)
```

**Description:**
- Calculates GFLOPs (Giga Floating Point Operations) for each model
- Plots GFLOPs vs. accuracy trade-offs
- Saves results as JSON and visualization plots
- Helps compare computational efficiency across methods

**Output:**
- Plot showing GFLOPs vs. validation accuracy
- JSON file with detailed metrics

---

### `mixture_model.py`
Creates and evaluates ensemble models by mixing multiple trained models.

**Usage:**
```bash
python scripts/evaluation/mixture_model.py
```

**Configuration:**
```python
sources = [
    "video_coop/kinetics400/k-16-ucf-attention-1",
    "dual_coop/kinetics400/k-16-ucf-attention-1"
]

eval_config = {
    "model_name": "coop",
    "dataset-train": "kinetics400",
    "dataset-test": "kinetics400",
    "split": "test",
    "run_id": "mixture_v1",
    "use-augmentation": False,
}
```

**Description:**
- Combines predictions from multiple models
- Can use equal or custom weights for each model
- Useful for ensemble learning experiments
- Evaluates mixture performance vs. individual models

## Common Parameters

### Model Keys Format
Model keys follow the pattern: `method/dataset/run_id`
- `method`: video_coop, dual_coop, stt, vilt, vita, vidop
- `dataset`: ucf101, hmdb51, kinetics400
- `run_id`: Specific identifier (e.g., "best-attention-0", "k-16")

### Datasets
Supported evaluation datasets:
- `ucf101`: UCF-101 (101 action classes)
- `hmdb51`: HMDB-51 (51 action classes)  
- `kinetics400`: Kinetics-400 (400 action classes)

### Batch Size and Workers
- Adjust `batch-size` based on GPU memory
- Increase `num-workers` for faster data loading (recommended: 4-8)

## Output

Evaluation results are typically saved to:
- `data/processed/results/` - Result JSON files
- WandB (if enabled) - Online logging and visualization
- Console - Real-time accuracy and metrics

## Notes

- All scripts automatically handle working directory setup
- Models must be trained before evaluation
- Checkpoints are loaded from configured paths
- Some scripts support WandB integration for experiment tracking
