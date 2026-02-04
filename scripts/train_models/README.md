# Model Training

This folder contains training scripts for different prompt tuning methods applied to CLIP-based video action recognition.

## Scripts

### `run_video_coop.py`
Trains Video-CoOp (Video Conditional Context Optimization) models.

**Usage:**
```bash
python scripts/train_models/run_video_coop.py
```

**Configuration:**
```python
fixed_config = {
    "dataset-config": {
        "batch-size": 4,           # Batch size per GPU
        "num-workers": 4,           # Data loading workers
        "batches-per-backprop": 32, # Gradient accumulation
        "K-train": None,            # Samples per class (None=all)
        "K-val": 4,                 # Validation samples per class
    },
    "continue-from": None,          # Checkpoint to resume from
    "use-augmentation": False,      # Training augmentation
    "temporal-pooling": "attention", # How to aggregate frames
    "ctx-len": 16,                  # Number of learnable context tokens
}
```

**Description:**
- Learns video-specific prompt tokens for CLIP
- Optimizes prompts on video-level context
- Supports attention-based temporal pooling
- Can continue training from existing checkpoints

---

### `run_dual_coop.py`
Trains Dual-CoOp models that learn both text and visual prompts.

**Usage:**
```bash
python scripts/train_models/run_dual_coop.py
```

**Configuration:**
```python
fixed_config = {
    "dataset-config": {
        "batch-size": 4,
        "num-workers": 2,
        "batches-per-backprop": 32,
        "K-train": None,
        "K-val": 4,
    },
    "continue-from": None,          # Or "dual_coop/kinetics400/full-ep-2"
    "use-augmentation": False,
    "videomix-type": None,          # Optional: "spatial", "temporal"
    "temporal-pooling": "attention", # "mean" or "attention"
    "num-temporal-views": 4,        # Number of temporal clips
}
```

**Description:**
- Jointly optimizes text and visual prompts
- Dual-stream prompt learning for better adaptation
- Supports VideoMix augmentation strategies
- Multiple temporal views for robust learning

---

### `run_stt.py`
Trains Spatial-Temporal Transformer (STT) models.

**Usage:**
```bash
python scripts/train_models/run_stt.py
```

**Configuration:**
```python
fixed_config = {
    "continue-from": None,
    "use-augmentation": False,
    "temporal-pooling": "mean",     # Typically uses mean pooling
    "videomix-type": "spatial",     # VideoMix augmentation type
    "dataset-config": {
        "K-train": None,
        "batch-size": 4,
        "num-workers": 2,
        "batches-per-backprop": 32,
    }
}
```

**Description:**
- Spatial-temporal transformer architecture
- Processes video sequences with transformer blocks
- Supports spatial VideoMix augmentation
- Good for capturing long-range temporal dependencies

---

### `run_vidop.py`
Trains ViDOP (Video Descriptor-based Optimal Prompting) models.

**Usage:**
```bash
python scripts/train_models/run_vidop.py
```

**Configuration:**
```python
fixed_config = {
    "temporal-pooling": "attention",
    "use-handcrafted-features": False,  # Use learned vs. hand-crafted
    "use-augmentation": True,           # Training augmentation
}
```

**Description:**
- Uses video descriptors for optimal prompt selection
- Can incorporate handcrafted features
- Attention-based temporal aggregation
- Designed for efficient prompt optimization

---

### `run_vilt.py`
Trains ViLT (Vision-and-Language Transformer) models.

**Usage:**
```bash
python scripts/train_models/run_vilt.py
```

**Description:**
- Vision-language transformer approach
- Joint processing of visual and textual information
- Unified transformer architecture

---

### `run_vita.py`
Trains ViTA (Video-Text Alignment) models.

**Usage:**
```bash
python scripts/train_models/run_vita.py
```

**Description:**
- Focuses on video-text alignment
- Optimizes for cross-modal matching
- Temporal video understanding

## Common Parameters

### Dataset Configuration
```python
"dataset-config": {
    "batch-size": 4,           # Batch size per GPU (reduce if OOM)
    "num-workers": 2-8,        # Parallel data loading workers
    "batches-per-backprop": 32, # Gradient accumulation steps
    "K-train": None,           # Samples per class (None=full dataset)
    "K-val": 4,                # Validation samples per class (for speed)
}
```

### Training Options
- `continue-from`: Resume from checkpoint (format: "method/dataset/run_id")
- `use-augmentation`: Apply training augmentations (color jitter, random crop, etc.)
- `temporal-pooling`: How to aggregate frame features
  - `"mean"`: Average pooling across time
  - `"attention"`: Learnable attention weights
- `videomix-type`: Augmentation strategy
  - `None`: No VideoMix
  - `"spatial"`: Mix spatial regions
  - `"temporal"`: Mix temporal segments

### Context Length
- `ctx-len`: Number of learnable prompt tokens (typically 4, 8, or 16)
- More tokens = more expressiveness but slower training

## Datasets

Supported datasets (set via `dataset_default`):
- `ucf101`: UCF-101 (101 action classes, ~13k videos)
- `hmdb51`: HMDB-51 (51 action classes, ~7k videos)
- `kinetics400`: Kinetics-400 (400 classes, ~240k training videos)

## Training Strategies

### Full Dataset Training
```python
"K-train": None  # Use all training samples
```

### Few-Shot Learning
```python
"K-train": 16    # 16 samples per class
"K-val": 4       # 4 samples per class for validation
```

### Transfer Learning
```python
"continue-from": "video_coop/kinetics400/full-ep-8"  # Pre-train on Kinetics
# Then fine-tune on target dataset (e.g., UCF-101)
```

### Gradient Accumulation
When GPU memory is limited:
```python
"batch-size": 4              # Small batch per step
"batches-per-backprop": 32   # Accumulate over 32 batches
# Effective batch size = 4 × 32 = 128
```

## Output

Training artifacts are saved to model checkpoints:
```
models/
├── {method_name}/
│   ├── {dataset_name}/
│   │   ├── {run_id}/
│   │   │   ├── checkpoint.pth       # Model weights
│   │   │   ├── config.json          # Training configuration
│   │   │   ├── training_log.json    # Training metrics
│   │   │   └── validation_results.json
```

## Monitoring

### WandB Integration
Training is automatically logged to Weights & Biases:
- Loss curves (training and validation)
- Accuracy metrics
- Learning rate schedules
- System metrics (GPU usage, etc.)

### Console Output
- Epoch progress bars
- Batch-level loss
- Validation accuracy after each epoch

## Tips

### Memory Management
- Reduce `batch-size` if you encounter OOM errors
- Increase `batches-per-backprop` to maintain effective batch size
- Reduce `num-workers` if you have limited CPU

### Training Speed
- Use more `num-workers` for faster data loading (4-8 recommended)
- Enable `use-augmentation` for better generalization
- Start with fewer epochs for quick experiments

### Hyperparameter Tuning
- Try different `ctx-len` values (4, 8, 16)
- Experiment with `temporal-pooling` methods
- Test with and without augmentation
- Adjust learning rate in base configs

### Continuing Training
To resume from a checkpoint:
```python
"continue-from": "video_coop/ucf101/best-attention-0"
```

## Common Issues

**Out of Memory (OOM)**
- Reduce batch size
- Reduce number of frames
- Use gradient checkpointing (if implemented)

**Slow Training**
- Increase num_workers
- Check data loading bottleneck
- Use smaller validation set (K-val)

**Poor Performance**
- Try different temporal pooling
- Enable augmentation
- Increase context length
- Pre-train on larger dataset

## Notes

- All scripts automatically handle working directory setup
- Training requires configured paths in `src/configs/paths.py`
- WandB token should be set in `tokens/wandb.txt`
- Models are saved automatically at best validation accuracy
- Early stopping is typically enabled to prevent overfitting
