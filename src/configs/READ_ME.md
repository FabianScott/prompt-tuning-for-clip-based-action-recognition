# Configuration Documentation

This directory contains JSON configuration files for datasets and methods used in the project. Most parameters are passed directly to the training pipeline, but some require additional explanation.

## Directory Structure

```
configs/
├── datasets/              # Dataset-specific configurations
│   ├── base.json         # Default dataset parameters
│   ├── ucf101.json       # UCF-101 dataset config
│   ├── hmdb51.json       # HMDB-51 dataset config
│   ├── kinetics400.json  # Kinetics-400 dataset config
│   └── augmentations-dict.json  # Augmentation strategies
├── methods/              # Method-specific configurations
│   ├── base.json         # Default training parameters
│   ├── video_coop.json   # Video-CoOp configuration
│   ├── dual_coop.json    # Dual-CoOp configuration
│   ├── stt.json          # STT configuration
│   ├── vidop.json        # ViDOP configuration
│   ├── vilt.json         # ViLT configuration
│   ├── vita.json         # ViTA configuration
│   └── checkpoint_paths.json  # Saved model paths
└── READ_ME.md            # This file
```

## Common Configuration Parameters

### Checkpoint Loading: `continue-from`

The `continue-from` parameter supports two formats for loading pre-trained models:

#### 1. Direct File Path
Provide an absolute or relative path to a PyTorch checkpoint file:
```json
{
  "continue-from": "/path/to/model/checkpoint.pt"
}
```

#### 2. Key-based Reference
Reference a checkpoint using keys defined in `checkpoint_paths.json`:
```json
{
  "continue-from": "video_coop/ucf101/best-attention-1"
}
```

**Key Format:** `{model_name}/{dataset}/{run_id}`
- `model_name`: Method name (e.g., "video_coop", "dual_coop", "stt")
- `dataset`: Training dataset (e.g., "ucf101", "hmdb51", "kinetics400")
- `run_id`: Specific run identifier (e.g., "best-attention-0", "k-16")

**Examples:**
- `"video_coop/ucf101/best-attention-0"` - Best Video-CoOp model on UCF-101 with attention pooling
- `"dual_coop/kinetics400/k-16-ucf-attention-1"` - Dual-CoOp pre-trained on Kinetics-400 with 16 shots
- `"stt/ucf101/best-true-val"` - Best STT model validated on true validation set

**Behavior:**
- If the key exists in `checkpoint_paths.json`, the corresponding checkpoint is loaded
- If the file path is valid, the checkpoint is loaded directly
- If neither is valid, training starts from scratch without pre-trained weights
- Useful for transfer learning and fine-tuning scenarios

---

### Few-Shot Learning: `K-train` and `K-val`

Controls the number of samples per class for training and validation.

**Options:**
- **Integer value** (e.g., `16`, `8`, `4`, `1`): Use K-shot learning with specified samples per class
  ```json
  {
    "K-train": 16,  // 16 samples per class for training
    "K-val": 4      // 4 samples per class for validation
  }
  ```

- **`null` or `None`**: Use the full dataset
  ```json
  {
    "K-train": null,  // Use all available training samples
    "K-val": null     // Use all available validation samples
  }
  ```

**Behavior:**
- When set to `null` and a validation folder exists, the entire dataset split is used
- No class-based sampling occurs with `null` values
- Useful for comparing few-shot vs. full-dataset performance
- Validation K-shot can be smaller than training for faster evaluation

**Common Scenarios:**
- **Full training:** `K-train: null, K-val: null`
- **16-shot learning:** `K-train: 16, K-val: 4`
- **1-shot learning:** `K-train: 1, K-val: 1`
- **Fast validation:** `K-train: null, K-val: 4` (full training, quick validation)

---

## Method-Specific Parameters

### ViLT (Vision-Language Transformer)

ViLT uses cross-modal transfer learning with specialized parameters:

#### `modality-transfer-text` and `modality-transfer-vision`

Defines which context tokens each transformer layer receives from text and vision modalities.

**Format:**
```json
{
  "modality-transfer-text": [
    [1, 0],  // Layer 0: text contexts only
    [1, 1],  // Layer 1: both text and vision contexts
    [1, 1],  // Layer 2: both text and vision contexts
    ...
  ],
  "modality-transfer-vision": [
    [0, 1],  // Layer 0: vision contexts only
    [1, 1],  // Layer 1: both text and vision contexts
    [1, 1],  // Layer 2: both text and vision contexts
    ...
  ]
}
```

**Structure:**
- **Outer list:** Each element corresponds to a transformer layer (index = layer index)
- **Inner list:** `[text_contexts, vision_contexts]`
  - First element (`0` or `1`): Whether to include **text contexts**
  - Second element (`0` or `1`): Whether to include **vision contexts**

**Context Transfer:**
- `[1, 0]`: Only text contexts (text-only layer)
- `[0, 1]`: Only vision contexts (vision-only layer)
- `[1, 1]`: Both text and vision contexts (cross-modal layer)
- `[0, 0]`: No contexts (not typically used)

**Translation Layer:**
When transferring from one modality to another (e.g., vision contexts to text layer), a translation/projection layer is automatically applied to align feature spaces.

**Example Configuration:**
```json
{
  "modality-transfer-text": [
    [1, 0],  // Early layers: text-specific
    [1, 0],
    [1, 1],  // Mid layers: cross-modal fusion
    [1, 1],
    [1, 1],
    [1, 1]   // Late layers: cross-modal reasoning
  ],
  "modality-transfer-vision": [
    [0, 1],  // Early layers: vision-specific
    [0, 1],
    [1, 1],  // Mid layers: cross-modal fusion
    [1, 1],
    [1, 1],
    [1, 1]   // Late layers: cross-modal reasoning
  ]
}
```

This configuration allows early layers to process modality-specific features, while later layers perform cross-modal reasoning.

---

## Common Training Parameters

### Learning Rate: `lr`
- Typical range: `1e-5` to `1e-3`
- Higher for prompt tuning (e.g., `8e-4`), lower for full fine-tuning (e.g., `2e-5`)

### Weight Decay: `weight-decay`
- L2 regularization strength
- Typical values: `0.001` to `0.01`

### Temporal Pooling: `temporal-pooling`
- `"mean"`: Average pooling across frames
- `"attention"`: Learnable attention-weighted pooling
- `"max"`: Max pooling (rarely used)

### Number of Frames: `num-frames`
- How many frames to sample per video
- Typical values: `8`, `16`, `32`
- More frames = better temporal understanding but slower training

### Augmentation: `use-augmentation`
- `true`: Apply training augmentations (recommended for better generalization)
- `false`: No augmentation (faster training, may overfit)

### VideoMix: `videomix-type`
- `null`: No VideoMix augmentation
- `"spatial"`: Mix spatial regions between videos
- `"temporal"`: Mix temporal segments between videos
- Useful for improving robustness

---

## Dataset Configuration Parameters

### Class Protocol: `class-protocol`
- `"all"`: Use all available classes
- `"base"`: Use base classes only (for base-novel splits)
- `"novel"`: Use novel classes only (for zero-shot evaluation)

### Split: `split`
- `"train"`: Training split
- `"val"`: Validation split
- `"test"`: Test split

### Video Sampling
- `num-temporal-views`: Number of temporal clips sampled per video
- `num-spatial-views`: Number of spatial crops per clip
- `space-between-frames`: Frame sampling stride (null = automatic)

---

## Tips and Best Practices

### Transfer Learning
1. Pre-train on large dataset (e.g., Kinetics-400):
   ```json
   {"continue-from": null, "K-train": null}
   ```
2. Fine-tune on target dataset (e.g., UCF-101):
   ```json
   {"continue-from": "video_coop/kinetics400/full-ep-8", "K-train": null}
   ```

### Few-Shot Learning
- Start with larger K (e.g., 16-shot) to verify setup
- Gradually reduce to test low-data regime (4-shot, 1-shot)
- Use smaller `K-val` for faster iterations

### Memory Management
- Reduce `num-frames` if GPU memory is limited
- Use gradient accumulation (`batches-per-backprop`) instead of large batch sizes
- Enable `fp16` for mixed precision training

### Debugging
- Set `debug: true` for verbose logging
- Use `use-profiling: true` for performance analysis
- Start with small `K-train` and `K-val` for quick experiments

---

## Example Configurations

### Full Training on UCF-101
```json
{
  "method-name": "video_coop",
  "continue-from": null,
  "K-train": null,
  "K-val": null,
  "use-augmentation": true,
  "temporal-pooling": "attention",
  "lr": 8e-4,
  "epochs": 40
}
```

### 16-Shot Learning on HMDB-51
```json
{
  "method-name": "dual_coop",
  "continue-from": null,
  "K-train": 16,
  "K-val": 4,
  "use-augmentation": false,
  "temporal-pooling": "attention",
  "lr": 8e-4,
  "epochs": 50
}
```

### Transfer Learning (Kinetics → UCF-101)
```json
{
  "method-name": "video_coop",
  "continue-from": "video_coop/kinetics400/full-ep-8",
  "K-train": null,
  "K-val": 4,
  "use-augmentation": true,
  "temporal-pooling": "attention",
  "lr": 2e-4,
  "epochs": 20
}
```

---

## Modifying Configurations

### Programmatically
```python
from src.configs.read_config import read_method_config

# Load base config
config = read_method_config("video_coop")

# Override parameters
config["lr"] = 1e-3
config["K-train"] = 16
config["continue-from"] = "video_coop/ucf101/best-attention-0"
```

### In Training Scripts
```python
fixed_config = {
    "continue-from": "video_coop/kinetics400/full-ep-8",
    "use-augmentation": True,
    "temporal-pooling": "attention",
    "dataset-config": {
        "K-train": 16,
        "K-val": 4,
    }
}

train_model(
    method_default="video_coop",
    dataset_default="ucf101",
    fixed_config=fixed_config
)
```

---

## Related Files

- **`parse_run_args.py`**: Parses and validates configuration parameters
- **`read_config.py`**: Loads and merges configuration files
- **`paths.py`**: Defines data and checkpoint paths
- **`checkpoint_paths.json`**: Maps checkpoint keys to file paths

For more details on specific methods, see the individual JSON files in the `methods/` directory.