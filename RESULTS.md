# Prompt Tuning for CLIP-Based Action Recognition - Results

**Last Updated:** 2026-02-12 10:14:00

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

| Model | Train | Val | Test | Hours |
| --- | --- | --- | --- | --- |
| VidOp (mean) | 52.1 | 54.4 | 51.4 | 4 |
| VidOp H (mean) | 63.9 | 52.5 | 51.4 | 4 |
| VidOp (attention 0) | 87.9 | 84.3 | 82.9 | 7 |
| VidOp H (attention 0) | 81.6 | 58.0 | 60.7 | 5 |
| CoOp (mean) | 88.6 | 86.8 | 85.6 | 6.5 |
| CoOp (attention 0) | 99.7 | 93.3 | 91.9 | 2 |
| CoOp (attention 1) | 100.0 | 93.5 | 92.5 | 8 |
| CoOp (attention 2) | 100.0 | 92.6 | 91.7 | 8 |
| Dual (mean) | 95.2 | 93.3 | 91.7 | 8.5 |
| Dual (attention 0) | 100.0 | 93.2 | 91.3 | 7 |
| Dual (attention 1) | 100.0 | 95.0 | 93.2 | 10 |
| Dual (attention 2) | 97.6 | 95.0 | 90.7 | 11 |
| ViLt (mean) | 93.6 | 86.5 | 84.8 | 18 |
| Vita (mean) | 99.3 | 91.0 | 90.8 | 15 |
| STT (mean) | 99.9 | 94.9 | 92.0 | 14 |

### VideoMix Augmentation Results

Results for models trained using the VideoMix data augmentation technique:

| Model | Train | Val | Test | Hours |
| --- | --- | --- | --- | --- |
| Dual-VideoMix (attention 0) | - | 92.3 | 91.3 | 7 |
| ViLt-VideoMix (mean) | - | 87.0 | 85.1 | 18 |
| Vita-VideoMix (mean) | - | 90.8 | 86.8 | 15 |
| STT-VideoMix (mean) | - | 91.5 | 91.5 | 14 |

### Temporal View Analysis

Impact of using single vs. multiple temporal views:

| Model | Test (1 temporal) | Delta |
| --- | --- | --- |
| VidOp (attention 0) | 81.8 | -1.1 |
| CoOp (mean) | 85.3 | -0.3 |
| CoOp (attention 0) | 90.0 | -1.9 |
| CoOp (attention 1) | 91.0 | -1.5 |
| CoOp (attention 2) | 89.5 | -2.2 |
| Dual (mean) | 91.8 | +0.1 |
| Dual (attention 0) | 90.7 | -0.6 |
| Dual (attention 1) | 88.9 | -4.3 |
| Dual (attention 2) | 92.3 | +1.6 |
| ViLt | 83.7 | -1.1 |
| Vita | 89.5 |  |
| STT | 90.0 | -2.0 |
| Dual-VideoMix (attention 0) | 88.8 | -2.5 |
| ViLt-VideoMix | 82.0 | -3.1 |
| ViTa-VideoMix | 86.7 | -0.1 |
| STT-VideoMix | 89.2 | -2.3 |

### Augmentation Robustness

#### Combined Augmentations

| Model | Test (Combined Aug) |
| --- | --- |
| VidOp (attention 0) | 64.9 |
| CoOp (mean) | 69.2 |
| CoOp (attention 0) | 69.0 |
| CoOp (attention 1) | 72.2 |
| CoOp (attention 2) | 64.9 |
| Dual (mean) | 75.7 |
| Dual (attention 0) | 73.6 |
| Dual-VideoMix (attention 0) | 76.6 |
| Dual (attention 1) | 73.1 |
| Dual (attention 2) | 68.5 |
| ViLt (mean) | 47.9 |
| Vita (mean) | 52.8 |
| STT (mean) | 69.4 |

#### Individual Augmentation Effects

| Model | H-Flip | Colour Jitter | Blur | S&P | Cutout p=0.1 | Cutout p=1 |
| --- | --- | --- | --- | --- | --- | --- |
| VidOp (attention 0) | 82.2 | 35.4 | 78.7 | 65.0 | 82.4 | 73.1 |
| CoOp (mean) | 85.0 | 44.4 | 68.2 | 82.4 | 85.3 | 77.5 |
| CoOp (attention 0) | 90.8 | 40.2 | 86.8 | 67.5 | 90.8 | 78.2 |
| CoOp (attention 1) | 91.3 | 42.7 | 87.4 | 70.9 | 91.3 | 81.6 |
| CoOp (attention 2) | 90.7 | 40.3 | 85.1 | 64.1 | 90.5 | 80.2 |
| Dual (mean) | 91.9 | 48.5 | 90.1 | 74.9 | 92.0 | 82.6 |
| Dual (attention 0) | 91.2 | 41.1 | 87.9 | 71.5 | 91.1 | 78.8 |
| Dual (attention 1) | 90.8 | 47.9 | 88.9 | 73.6 | 90.5 | 79.4 |
| Dual (attention 2) | 92.7 | 43.9 | 90.2 | 67.6 | 92.7 | 82.1 |
| ViLt | 83.9 | 32.3 | 80.7 | 46.7 | 83.8 | 69.0 |
| ViTa | 89.6 | 31.7 | 79.1 | 52.4 | 89.6 | 78.4 |
| STT | 90.3 | 38.8 | 82.4 | 70.0 | 90.3 | 81.0 |
| Dual-VideoMix (attention 0) | 90.1 | 48.6 | 85.2 | 75.6 | 90.1 | 85.6 |
| ViLt-VideoMix | 83.0 | 27.5 | 80.0 | 70.1 | 83.1 | 83.0 |
| ViTa-VideoMix | 86.6 | 40.5 | 84.2 | 68.1 | 86.7 | 83.2 |
| STT-VideoMix | 90.7 | 43.0 | 84.6 | 71.1 | 90.5 | 84.7 |

#### Training with Augmentations

| Model | Train | Val | Test | Hours |
| --- | --- | --- | --- | --- |
| VidOp (attention 0) | 81.0 | 75.3 | 72.7 | 7 |
| CoOp (mean) | 78.0 | 76.6 | 85.6 | 8 |
| CoOp (attention 0) | 94.4 | 82.2 | 77.5 | 4 |
| Dual (mean) | 80.8 | 84.0 | 76.4 | 6.5 |
| Dual (attention 0) | 94.4 | 82.2 | 84.0 | 7 |
| ViLt | 93.6 | 86.5 | 76.7 | 21 |
| ViTa | 92.9 | 82.8 | 81.4 | 13 |
| STT | 98.7 | 85.8 | 82.9 | 21 |

### Class-Level Performance Analysis

#### Classes Below Threshold Per Model

**Number of classes below each accuracy threshold per model across UCF101 test set**

| Threshold | Co a0 | Co a1 | Co a2 | Du M | Du a0 | Du a1 | Du a2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50% | 2 | 1 | 1 | 1 | 1 | 1 | 3 |
| 60% | 5 | 5 | 5 | 4 | 4 | 4 | 4 |
| 70% | 8 | 9 | 11 | 9 | 10 | 7 | 8 |
| 80% | 13 | 13 | 15 | 14 | 14 | 11 | 18 |
| 85% | 21 | 16 | 21 | 19 | 24 | 16 | 24 |
| 90% | 27 | 26 | 31 | 30 | 30 | 23 | 36 |
| 95% | 41 | 39 | 39 | 44 | 37 | 34 | 42 |

#### Overlapping Difficult Classes

**Number of overlapping difficult classes by minimum model agreement**

| Threshold | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50% | 5 | 3 | 2 | 0 | 0 | 0 | 0 |
| 60% | 10 | 7 | 6 | 5 | 2 | 1 | 0 |
| 70% | 17 | 11 | 11 | 8 | 7 | 5 | 3 |
| 80% | 24 | 18 | 13 | 12 | 12 | 10 | 9 |
| 85% | 36 | 29 | 21 | 17 | 15 | 12 | 11 |
| 90% | 44 | 37 | 34 | 31 | 24 | 18 | 15 |
| 95% | 56 | 46 | 42 | 40 | 36 | 31 | 25 |

### Black Strip Analysis

Impact of removing black strips from videos:

| Model | Test (No Black Strips) |
| --- | --- |
| VidOp (attention 0) | 77.0 |
| CoOp (mean) | 85.0 |
| CoOp (attention 0) | 90.6 |
| CoOp (attention 1) | 91.6 |
| CoOp (attention 2) | 91.1 |
| Dual (mean) | 80.5 |
| Dual (attention 0) | 89.3 |
| Dual-VideoMix (attention 0) | 84.4 |
| Dual (attention 1) | 82.1 |
| Dual (attention 2) | 72.1 |
| ViLt | 80.9 |
| ViTa | 89.4 |
| STT | 90.1 |

---

## Kinetics400 Results

### Zero-Shot Transfer

Transfer learning from UCF101 to Kinetics400 without fine-tuning:

| Model | Top-1 Accuracy |
| --- | --- |
| VidOp (attention 0) | 9.0 |
| CoOp (mean) | 15.6 |
| CoOp (attention 0) | 5.0 |
| CoOp (attention 1) | 10.7 |
| CoOp (attention 2) | 9.9 |
| Dual (mean) | 19.5 |
| Dual (attention 0) | 3.7 |
| Dual-VideoMix (attention 0) | 3.9 |
| Dual (attention 1) | 13.1 |
| Dual (attention 2) | 12.2 |
| ViLt (mean) | 7.7 |
| Vita (mean) | 2.7 |
| STT (mean) | 10.5 |
| ViLt-VideoMix | 5.5 |
| ViTa-VideoMix | 7.9 |
| STT-VideoMix | 10.9 |

### K-Shot Learning

Few-shot learning on Kinetics400 (4-shot and 16-shot):

| Model | 16-shot Val | 16-shot Test | 4-shot Val | 4-shot Test |
| --- | --- | --- | --- | --- |
| CoOp (attention 0) | 57.1 | 54.6 | 42.8 | 23.1 |
| CoOp T (attention 1) | 61.7 | 59.9 | 50.4 | 23.1 |
| Dual (attention 0) | 63.9 | 60.3 | 43.8 | 42.0 |
| Dual T (attention 1) | 63.3 | 61.9 | 55.6 | 58.7 |
| CoOp+Dual T Mixture | - | 63.7 | - | - |
| STT T | 22.5 | - | - | - |
| Vita T | 13.2 | 13.0 | - | - |
| ViLT T | 17.9 | 17.4 | - | - |

---

## HMDB51 Results

### Zero-Shot Transfer

Transfer learning from UCF101 to HMDB51 without fine-tuning:

| Model | Top-1 Accuracy |
| --- | --- |
| VidOp (attention 0) | 19.6 |
| CoOp (mean) | 21.9 |
| CoOp (attention 0) | 10.2 |
| CoOp (attention 1) | 25.2 |
| CoOp (attention 2) | 24.4 |
| Dual (mean) | 21.9 |
| Dual (attention 0) | 9.9 |
| Dual-VideoMix (attention 0) | 7.7 |
| Dual (attention 1) | 22.3 |
| Dual (attention 2) | 22.2 |
| ViLt (mean) | 18.6 |
| Vita (mean) | 7.9 |
| STT (mean) | 14.2 |
| ViLt-VideoMix | 11.8 |
| ViTa-VideoMix | 12.9 |
| STT-VideoMix | 11.2 |

### K-Shot Learning

Few-shot learning on HMDB51 (4-shot and 16-shot):

| Model | 16-shot Val | 16-shot Test | 4-shot Val | 4-shot Test |
| --- | --- | --- | --- | --- |
| CoOp (attention 0) | 63.2 | 67.6 | 24.7 | 21.7 |
| CoOp T (attention 1) | 69.0 | 67.6 | 27.8 | 28.9 |
| Dual (attention 0) | 68.4 | 60.6 | 46.9 | 43.5 |
| Dual T (attention 1) | 72.1 | 61.0 | 30.5 | 28.7 |
| ViLT T | 42.6 | 42.1 | 23.2 | 26.3 |
| Vita T | 62.6 | 54.6 | 29.4 | 27.4 |

---

## Computational Cost Analysis

### GFLOPs vs. Accuracy

The following figures show the relationship between computational cost (GFLOPs) and model accuracy.


#### Gflops Vs Val Accuracy 101 Classes

![Gflops Vs Val Accuracy 101 Classes](figures/gflops/gflops_vs_val_accuracy_101_classes.png)


#### Gflops Vs Num Classes

![Gflops Vs Num Classes](figures/gflops/gflops_vs_num_classes.png)


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


#### dual_coopbest-attention-1-ucf101-1-attention-rollout-cls

**ApplyEyeMakeup**

![ApplyEyeMakeup](figures/explainer/dual_coopbest-attention-1-ucf101-1-attention-rollout-cls/ApplyEyeMakeup/1-video0_view0.png)

**ApplyEyeMakeup**

![ApplyEyeMakeup](figures/explainer/dual_coopbest-attention-1-ucf101-1-attention-rollout-cls/ApplyEyeMakeup/1-video0_view1.png)

**BabyCrawling**

![BabyCrawling](figures/explainer/dual_coopbest-attention-1-ucf101-1-attention-rollout-cls/BabyCrawling/118-video0_view0.png)

**BabyCrawling**

![BabyCrawling](figures/explainer/dual_coopbest-attention-1-ucf101-1-attention-rollout-cls/BabyCrawling/118-video0_view1.png)


#### video_coopbest-attention-0-ucf101-1-attention-rollout-cls

**ApplyEyeMakeup**

![ApplyEyeMakeup](figures/explainer/video_coopbest-attention-0-ucf101-1-attention-rollout-cls/ApplyEyeMakeup/1-video0_view0.png)

**ApplyEyeMakeup**

![ApplyEyeMakeup](figures/explainer/video_coopbest-attention-0-ucf101-1-attention-rollout-cls/ApplyEyeMakeup/1-video0_view1.png)

**BabyCrawling**

![BabyCrawling](figures/explainer/video_coopbest-attention-0-ucf101-1-attention-rollout-cls/BabyCrawling/118-video0_view0.png)

**BabyCrawling**

![BabyCrawling](figures/explainer/video_coopbest-attention-0-ucf101-1-attention-rollout-cls/BabyCrawling/118-video0_view1.png)


#### sttbest-true-val-ucf101-1-attention-rollout-cls

**ApplyEyeMakeup**

![ApplyEyeMakeup](figures/explainer/sttbest-true-val-ucf101-1-attention-rollout-cls/ApplyEyeMakeup/1-video0_view0.png)

**ApplyEyeMakeup**

![ApplyEyeMakeup](figures/explainer/sttbest-true-val-ucf101-1-attention-rollout-cls/ApplyEyeMakeup/1-video0_view1.png)

**BabyCrawling**

![BabyCrawling](figures/explainer/sttbest-true-val-ucf101-1-attention-rollout-cls/BabyCrawling/118-video0_view0.png)

**BabyCrawling**

![BabyCrawling](figures/explainer/sttbest-true-val-ucf101-1-attention-rollout-cls/BabyCrawling/118-video0_view1.png)


### Key Findings

- Models attend to black strips when present in videos
- Attention patterns don't always align with human-interpretable features
- Removing black strips significantly impacts some models (Dual mean: -11%, Dual a-2: -18.6%)
- GradCAM returns all zeros due to gradient flow through classification token

---

## Calibration Results

Model calibration analysis showing confidence vs. accuracy:

<table>
<tr>
<td align='center'><b>dual_coop-ucf101-best-attention-0</b><br/><img src='figures/calibration/ucf101/dual_coop-ucf101-best-attention-0/2026-02-03_18-00-16/reliability_diagram.png' width='300'/></td>
<td align='center'><b>dual_coop-ucf101-best-attention-1</b><br/><img src='figures/calibration/ucf101/dual_coop-ucf101-best-attention-1/2026-02-03_18-17-56/reliability_diagram.png' width='300'/></td>
<td align='center'><b>dual_coop-ucf101-best-attention-2</b><br/><img src='figures/calibration/ucf101/dual_coop-ucf101-best-attention-2/2026-02-03_18-26-47/reliability_diagram.png' width='300'/></td>
</tr>
<tr>
<td align='center'><b>dual_coop-ucf101-best-mean</b><br/><img src='figures/calibration/ucf101/dual_coop-ucf101-best-mean/2026-02-03_17-51-26/reliability_diagram.png' width='300'/></td>
<td align='center'><b>dual_coop-ucf101-videomix-attention-0</b><br/><img src='figures/calibration/ucf101/dual_coop-ucf101-videomix-attention-0/2026-02-03_18-09-06/reliability_diagram.png' width='300'/></td>
<td align='center'><b>stt-ucf101-best-true-val</b><br/><img src='figures/calibration/ucf101/stt-ucf101-best-true-val/2026-02-03_18-53-44/reliability_diagram.png' width='300'/></td>
</tr>
<tr>
<td align='center'><b>stt-ucf101-videomix</b><br/><img src='figures/calibration/ucf101/stt-ucf101-videomix/2026-02-03_19-24-41/reliability_diagram.png' width='300'/></td>
<td align='center'><b>video_coop-ucf101-best-attention-0</b><br/><img src='figures/calibration/ucf101/video_coop-ucf101-best-attention-0/2026-02-03_17-25-02/reliability_diagram.png' width='300'/></td>
<td align='center'><b>video_coop-ucf101-best-attention-1</b><br/><img src='figures/calibration/ucf101/video_coop-ucf101-best-attention-1/2026-02-03_17-33-51/reliability_diagram.png' width='300'/></td>
</tr>
<tr>
<td align='center'><b>video_coop-ucf101-best-attention-2</b><br/><img src='figures/calibration/ucf101/video_coop-ucf101-best-attention-2/2026-02-03_17-42-39/reliability_diagram.png' width='300'/></td>
<td align='center'><b>video_coop-ucf101-best-mean</b><br/><img src='figures/calibration/ucf101/video_coop-ucf101-best-mean/2026-02-04_10-57-21/reliability_diagram.png' width='300'/></td>
<td align='center'><b>vidop-ucf101-best-no-hand-features-attention-0</b><br/><img src='figures/calibration/ucf101/vidop-ucf101-best-no-hand-features-attention-0/2026-02-04_10-48-24/reliability_diagram.png' width='300'/></td>
</tr>
<tr>
<td align='center'><b>vilt-ucf101-best-mean</b><br/><img src='figures/calibration/ucf101/vilt-ucf101-best-mean/2026-02-03_18-35-37/reliability_diagram.png' width='300'/></td>
<td align='center'><b>vilt-ucf101-videomix</b><br/><img src='figures/calibration/ucf101/vilt-ucf101-videomix/2026-02-03_19-06-33/reliability_diagram.png' width='300'/></td>
<td align='center'><b>vita-ucf101-ep-16</b><br/><img src='figures/calibration/ucf101/vita-ucf101-ep-16/2026-02-03_18-44-40/reliability_diagram.png' width='300'/></td>
</tr>
<tr>
<td align='center'><b>vita-ucf101-videomix</b><br/><img src='figures/calibration/ucf101/vita-ucf101-videomix/2026-02-03_19-15-38/reliability_diagram.png' width='300'/></td>
</tr>
</table>


---

## Model Configurations

All trained model configurations for UCF101 are available in the [`models/trained_configs`](models/trained_configs) directory.

### Configuration Files by Architecture


**DUAL_COOP** (7 configurations)

- [ucf101 (base)](models/configs/dual_coop/ucf101.json)
- [dual_coop_ucf101_best-attention-0](models/configs/dual_coop/dual_coop_ucf101_best-attention-0.json)
- [dual_coop_ucf101_best-attention-1](models/configs/dual_coop/dual_coop_ucf101_best-attention-1.json)
- [dual_coop_ucf101_best-attention-2](models/configs/dual_coop/dual_coop_ucf101_best-attention-2.json)
- [dual_coop_ucf101_best-mean](models/configs/dual_coop/dual_coop_ucf101_best-mean.json)
- [dual_coop_ucf101_videomix-attention-0](models/configs/dual_coop/dual_coop_ucf101_videomix-attention-0.json)
- [ucf101](models/configs/dual_coop/ucf101.json)


**STT** (4 configurations)

- [ucf101 (base)](models/configs/stt/ucf101.json)
- [stt_ucf101_best-mean](models/configs/stt/stt_ucf101_best-mean.json)
- [stt_ucf101_videomix](models/configs/stt/stt_ucf101_videomix.json)
- [ucf101](models/configs/stt/ucf101.json)


**VIDEO_COOP** (6 configurations)

- [ucf101 (base)](models/configs/video_coop/ucf101.json)
- [ucf101](models/configs/video_coop/ucf101.json)
- [video_coop_ucf101_best-attention-0](models/configs/video_coop/video_coop_ucf101_best-attention-0.json)
- [video_coop_ucf101_best-attention-1](models/configs/video_coop/video_coop_ucf101_best-attention-1.json)
- [video_coop_ucf101_best-attention-2](models/configs/video_coop/video_coop_ucf101_best-attention-2.json)
- [video_coop_ucf101_best-mean](models/configs/video_coop/video_coop_ucf101_best-mean.json)


**VIDOP** (6 configurations)

- [ucf101 (base)](models/configs/vidop/ucf101.json)
- [ucf101](models/configs/vidop/ucf101.json)
- [vidop_ucf101_best-no-hand-features-attention-0](models/configs/vidop/vidop_ucf101_best-no-hand-features-attention-0.json)
- [vidop_ucf101_best-no-hand-features-mean](models/configs/vidop/vidop_ucf101_best-no-hand-features-mean.json)
- [vidop_ucf101_best-with-hand-features-attention-0](models/configs/vidop/vidop_ucf101_best-with-hand-features-attention-0.json)
- [vidop_ucf101_best-with-hand-features-mean](models/configs/vidop/vidop_ucf101_best-with-hand-features-mean.json)


**VILT** (4 configurations)

- [ucf101 (base)](models/configs/vilt/ucf101.json)
- [ucf101](models/configs/vilt/ucf101.json)
- [vilt_ucf101_best-mean](models/configs/vilt/vilt_ucf101_best-mean.json)
- [vilt_ucf101_videomix](models/configs/vilt/vilt_ucf101_videomix.json)


**VITA** (4 configurations)

- [ucf101 (base)](models/configs/vita/ucf101.json)
- [ucf101](models/configs/vita/ucf101.json)
- [vita_ucf101_beat-mean](models/configs/vita/vita_ucf101_beat-mean.json)
- [vita_ucf101_videomix](models/configs/vita/vita_ucf101_videomix.json)


**Total Configurations**: 31 model and dataset configs for UCF101

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

All tables in this document are generated from scripts in `scripts/tables/`:

```bash
# Generate all tables
python scripts/tables/generate_all_tables.py

# Verify all tables are covered
python scripts/tables/verify_coverage.py

# Generate individual table
python scripts/tables/ucf101_main_results.py
```

See [scripts/tables/README.md](scripts/tables/README.md) for complete documentation.

---

**Note**: This README is automatically generated. To update with latest results:

```bash
python generate_results_readme.py
```
