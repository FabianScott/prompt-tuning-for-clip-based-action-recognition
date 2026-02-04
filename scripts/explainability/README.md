# Model Explainability

This folder contains scripts for analyzing and visualizing model predictions, including calibration analysis and attention visualization.

## Scripts

### `calibration.py`
Analyzes model calibration by computing expected calibration error (ECE) and generating reliability diagrams.

**Usage:**
```bash
python scripts/explainability/calibration.py
```

**Configuration:**
Edit the `model_keys` list with models to analyze:
```python
model_keys = [
    "video_coop/ucf101/best-attention-0",
    "dual_coop/ucf101/best-attention-1",
    "stt/ucf101/best-true-val",
]
```

**Description:**
- Evaluates how well predicted confidence scores match actual accuracy
- Generates reliability diagrams showing calibration
- Computes Expected Calibration Error (ECE) metrics
- Useful for understanding model confidence reliability

**Meta Config Options:**
- `batch-size`: Batch size for evaluation (default: 16)
- `num-workers`: Number of data loading workers (default: 8)
- `use-wandb`: Whether to log to WandB (default: True)
- `re-run`: Force re-evaluation even if results exist (default: True)

**Evaluation Config:**
- `model_name`: Model method to evaluate
- `dataset-train`: Dataset used for training the model
- `dataset-test`: Dataset to evaluate calibration on (default: "ucf101")
- `split`: Dataset split to use (default: "test")
- `use-augmentation`: Whether to use test-time augmentation (default: False)
- `use-handcrafted-prompts`: Use predefined prompts vs. learned prompts (default: False)

**Output:**
- Calibration plots saved to `figures/calibration/`
- ECE metrics logged to WandB or console
- Reliability diagrams for each model

---

### `run_explainer.py`
Generates visual explanations of model predictions using attention mechanisms and Grad-CAM.

**Usage:**
```bash
python scripts/explainability/run_explainer.py
```

**Configuration:**
Define a `grad_cam_config` dictionary:
```python
grad_cam_config = {
    "model-name": "video_coop",
    "dataset-train": "ucf101",
    "dataset-test": "ucf101",
    "run-id": "best-attention-0",
    "classes-to-map": 10,  # Number of classes to visualize
    "classes-to-mix": None,  # For VideoMix experiments
    "split": "test",
    "num-frames": 8,
    "explainer-method": "attention-rollout",  # or "grad-cam"
    "use-cls-token": True,
    "weighted-by-predictions": False,
}
```

**Description:**
- Visualizes which spatial and temporal regions the model focuses on
- Supports multiple explainability methods:
  - `attention-rollout`: Aggregates attention across layers
  - `grad-cam`: Gradient-based class activation mapping
- Generates heatmaps overlaid on input frames
- Can weight attention by prediction confidence

**Explainer Methods:**
- **Attention Rollout**: Traces attention flow through transformer layers
- **Grad-CAM**: Highlights discriminative regions for predictions

**Additional Options:**
- `use-cls-token`: Focus on CLS token attention (True) or average all tokens (False)
- `weighted-by-predictions`: Weight heatmaps by prediction confidence
- `num-frames`: Number of frames to visualize per video

**Output:**
- Attention heatmaps saved to `figures/explainer/`
- Videos with overlaid attention visualizations
- Per-sample and aggregated attention patterns
- Results logged to WandB (if enabled)

**Output Directory Structure:**
```
figures/explainer/
├── {model-key}-{method}/
│   ├── video_samples/
│   │   ├── class_X_sample_Y.mp4
│   │   └── ...
│   ├── attention_maps/
│   │   └── ...
│   └── aggregated_stats.json
```

## Use Cases

### Calibration Analysis
Use `calibration.py` to:
- Assess whether model confidence scores are reliable
- Compare calibration across different methods
- Identify over-confident or under-confident predictions
- Generate publication-ready calibration plots

### Visual Explanation
Use `run_explainer.py` to:
- Understand which video regions influence predictions
- Debug model errors by visualizing attention
- Compare what different models focus on
- Generate figures for papers and presentations
- Validate that models learn meaningful features

## Common Patterns

### Analyzing Multiple Models
Both scripts support batch processing:
```python
model_keys = [
    "video_coop/ucf101/best-attention-0",
    "dual_coop/ucf101/best-attention-1",
    "stt/ucf101/best-true-val",
]

for model_key in model_keys:
    # Run calibration or explainer
    ...
```

### Dataset Selection
- Use `dataset-test` to specify evaluation dataset
- Can differ from training dataset for transfer learning analysis
- Supports: ucf101, hmdb51, kinetics400

### WandB Integration
- Set `use-wandb: True` in meta_config to log results
- Visualizations and metrics are automatically uploaded
- Enables easy comparison across experiments

## Requirements

### For Calibration
- Trained model checkpoints
- Access to evaluation dataset
- Sufficient memory for model inference

### For Explainability
- Requires `wandb[media]` for video logging:
  ```bash
  pip install wandb[media]
  ```
- May require more GPU memory for gradient computation
- Processing time increases with number of samples

## Notes

- All scripts handle working directory setup automatically
- Model checkpoints must exist before running analysis
- Results are cached to avoid redundant computation
- Large batch sizes may cause memory issues with explainability methods
- Visualizations work best with 8-16 frames per video
