"""
Centralized results data for all experiments.
All table data is stored as Python dictionaries for easy access and generation.
"""

# UCF101 Main Results
# Format: [train_acc, val_acc, test_acc, train_hours]
UCF101_MAIN_RESULTS = {
    "VidOp (mean)": [52.1, 54.4, 51.4, 4],
    "VidOp H (mean)": [63.9, 52.5, 51.4, 4],
    "VidOp (attention 0)": [87.9, 84.3, 82.9, 7],
    "VidOp H (attention 0)": [81.6, 58.0, 60.7, 5],
    "CoOp (mean)": [88.6, 86.8, 85.6, 6.5],
    "CoOp (attention 0)": [99.7, 93.3, 91.9, 2],
    "CoOp (attention 1)": [100.0, 93.5, 92.5, 8],
    "CoOp (attention 2)": [100.0, 92.6, 91.7, 8],
    "Dual (mean)": [95.2, 93.3, 91.7, 8.5],
    "Dual (attention 0)": [100.0, 93.2, 91.3, 7],
    "Dual (attention 1)": [100.0, 95.0, 93.2, 10],
    "Dual (attention 2)": [97.6, 95.0, 90.7, 11],
    "ViLt (mean)": [93.6, 86.5, 84.8, 18],
    "Vita (mean)": [99.3, 91.0, 90.8, 15],
    "STT (mean)": [99.9, 94.9, 92.0, 14],
}

# UCF101 VideoMix Results
# Format: [train_acc, val_acc, test_acc, train_hours]
UCF101_VIDEOMIX_RESULTS = {
    "Dual-VideoMix (attention 0)": [None, 92.3, 91.3, 7],
    "ViLt-VideoMix (mean)": [None, 87.0, 85.1, 18],
    "Vita-VideoMix (mean)": [None, 90.8, 86.8, 15],
    "STT-VideoMix (mean)": [None, 91.5, 91.5, 14],
}

# UCF101 1 Temporal View Results
# Format: [test_acc]
UCF101_1_TEMPORAL_VIEW = {
    "VidOp (attention 0)": [81.8],
    "CoOp (mean)": [85.3],
    "CoOp (attention 0)": [90.0],
    "CoOp (attention 1)": [91.0],
    "CoOp (attention 2)": [89.5],
    "Dual (mean)": [91.8],
    "Dual (attention 0)": [90.7],
    "Dual (attention 1)": [88.9],
    "Dual (attention 2)": [92.3],
    "ViLt": [83.7],
    "Vita": [89.5],
    "STT": [90.0],
    "Dual-VideoMix (attention 0)": [88.8],
    "ViLt-VideoMix": [82.0],
    "ViTa-VideoMix": [86.7],
    "STT-VideoMix": [89.2],
}

# Baseline test accuracies for delta calculations
BASELINE_TEST = {
    "VidOp (attention 0)": 82.9,
    "CoOp (mean)": 85.6,
    "CoOp (attention 0)": 91.9,
    "CoOp (attention 1)": 92.5,
    "CoOp (attention 2)": 91.7,
    "Dual (mean)": 91.7,
    "Dual (attention 0)": 91.3,
    "Dual (attention 1)": 93.2,
    "Dual (attention 2)": 90.7,
    "ViLt": 84.8,
    "ViTa": 90.8,
    "STT": 92.0,
    "Dual-VideoMix (attention 0)": 91.3,
    "ViLt-VideoMix": 85.1,
    "ViTa-VideoMix": 86.8,
    "STT-VideoMix": 91.5,
}

# UCF101 Combined Augmentation Results
# Format: [test_acc]
UCF101_COMBINED_AUGMENTATION = {
    "VidOp (attention 0)": 64.9,
    "CoOp (mean)": 69.2,
    "CoOp (attention 0)": 69.0,
    "CoOp (attention 1)": 72.2,
    "CoOp (attention 2)": 64.9,
    "Dual (mean)": 75.7,
    "Dual (attention 0)": 73.6,
    "Dual-VideoMix (attention 0)": 76.6,
    "Dual (attention 1)": 73.1,
    "Dual (attention 2)": 68.5,
    "ViLt (mean)": 47.9,
    "Vita (mean)": 52.8,
    "STT (mean)": 69.4,
}

# UCF101 Individual Augmentations
# Format: [H-Flip, Colour Jitter, Blur, S&P, Cutout p=0.1, Cutout p=1]
UCF101_INDIVIDUAL_AUGMENTATIONS = {
    "VidOp (attention 0)": [82.2, 35.4, 78.7, 65.0, 82.4, 73.1],
    "CoOp (mean)": [85.0, 44.4, 68.2, 82.4, 85.3, 77.5],
    "CoOp (attention 0)": [90.8, 40.2, 86.8, 67.5, 90.8, 78.2],
    "CoOp (attention 1)": [91.3, 42.7, 87.4, 70.9, 91.3, 81.6],
    "CoOp (attention 2)": [90.7, 40.3, 85.1, 64.1, 90.5, 80.2],
    "Dual (mean)": [91.9, 48.5, 90.1, 74.9, 92.0, 82.6],
    "Dual (attention 0)": [91.2, 41.1, 87.9, 71.5, 91.1, 78.8],
    "Dual (attention 1)": [90.8, 47.9, 88.9, 73.6, 90.5, 79.4],
    "Dual (attention 2)": [92.7, 43.9, 90.2, 67.6, 92.7, 82.1],
    "ViLt": [83.9, 32.3, 80.7, 46.7, 83.8, 69.0],
    "ViTa": [89.6, 31.7, 79.1, 52.4, 89.6, 78.4],
    "STT": [90.3, 38.8, 82.4, 70.0, 90.3, 81.0],
    "Dual-VideoMix (attention 0)": [90.1, 48.6, 85.2, 75.6, 90.1, 85.6],
    "ViLt-VideoMix": [83.0, 27.5, 80.0, 70.1, 83.1, 83.0],
    "ViTa-VideoMix": [86.6, 40.5, 84.2, 68.1, 86.7, 83.2],
    "STT-VideoMix": [90.7, 43.0, 84.6, 71.1, 90.5, 84.7],
}

# UCF101 Training with Augmentations
# Format: [train_acc, val_acc, test_acc, train_hours]
UCF101_TRAIN_AUGMENTED = {
    "VidOp (attention 0)": [81.0, 75.3, 72.7, 7],
    "CoOp (mean)": [78.0, 76.6, 85.6, 8],
    "CoOp (attention 0)": [94.4, 82.2, 77.5, 4],
    "Dual (mean)": [80.8, 84.0, 76.4, 6.5],
    "Dual (attention 0)": [94.4, 82.2, 84.0, 7],
    "ViLt": [93.6, 86.5, 76.7, 21],
    "ViTa": [92.9, 82.8, 81.4, 13],
    "STT": [98.7, 85.8, 82.9, 21],
}

# UCF101 Removed Black Strips
# Format: [test_acc]
UCF101_REMOVED_BLACK_STRIPS = {
    "VidOp (attention 0)": [77.0],
    "CoOp (mean)": [85.0],
    "CoOp (attention 0)": [90.6],
    "CoOp (attention 1)": [91.6],
    "CoOp (attention 2)": [91.1],
    "Dual (mean)": [80.5],
    "Dual (attention 0)": [89.3],
    "Dual-VideoMix (attention 0)": [84.4],
    "Dual (attention 1)": [82.1],
    "Dual (attention 2)": [72.1],
    "ViLt": [80.9],
    "ViTa": [89.4],
    "STT": [90.1],
}

# Distribution Tables
CLASSES_UNDER_THRESHOLD = {
    "thresholds": [50, 60, 70, 80, 85, 90, 95],
    "models": {
        "Co a0": [2, 5, 8, 13, 21, 27, 41],
        "Co a1": [1, 5, 9, 13, 16, 26, 39],
        "Co a2": [1, 5, 11, 15, 21, 31, 39],
        "Du M": [1, 4, 9, 14, 19, 30, 44],
        "Du a0": [1, 4, 10, 14, 24, 30, 37],
        "Du a1": [1, 4, 7, 11, 16, 23, 34],
        "Du a2": [3, 4, 8, 18, 24, 36, 42],
    }
}

OVERLAP_CLASSES_BY_THRESHOLD = {
    "thresholds": [50, 60, 70, 80, 85, 90, 95],
    "min_models": [1, 2, 3, 4, 5, 6, 7],
    "overlaps": [
        [5, 3, 2, 0, 0, 0, 0],  # 50%
        [10, 7, 6, 5, 2, 1, 0],  # 60%
        [17, 11, 11, 8, 7, 5, 3],  # 70%
        [24, 18, 13, 12, 12, 10, 9],  # 80%
        [36, 29, 21, 17, 15, 12, 11],  # 85%
        [44, 37, 34, 31, 24, 18, 15],  # 90%
        [56, 46, 42, 40, 36, 31, 25],  # 95%
    ]
}

# Kinetics400 Results
KINETICS400_ZERO_SHOT = {
    "VidOp (attention 0)": 9.0,
    "CoOp (mean)": 15.6,
    "CoOp (attention 0)": 5.0,
    "CoOp (attention 1)": 10.7,
    "CoOp (attention 2)": 9.9,
    "Dual (mean)": 19.5,
    "Dual (attention 0)": 3.7,
    "Dual-VideoMix (attention 0)": 3.9,
    "Dual (attention 1)": 13.1,
    "Dual (attention 2)": 12.2,
    "ViLt (mean)": 7.7,
    "Vita (mean)": 2.7,
    "STT (mean)": 10.5,
}

# Format: [16-shot-val, 16-shot-test, 4-shot-val, 4-shot-test]
KINETICS400_KSHOT = {
    "CoOp (attention 0)": [57.1, 54.6, 42.8, 23.1],
    "CoOp T (attention 1)": [61.7, 59.9, 50.4, 23.1],
    "Dual (attention 0)": [63.9, 60.3, 43.8, 42.0],
    "Dual T (attention 1)": [63.3, 61.9, 55.6, 58.7],
    "CoOp+Dual T Mixture": [None, 63.7, None, None],
    "STT T": [22.5, None, None, None],
    "Vita T": [13.2, 13.0, None, None],
    "ViLT T": [17.9, 17.4, None, None],
}

# HMDB51 Results
HMDB51_ZERO_SHOT = {
    "VidOp (attention 0)": 19.6,
    "CoOp (mean)": 21.9,
    "CoOp (attention 0)": 10.2,
    "CoOp (attention 1)": 25.2,
    "CoOp (attention 2)": 24.4,
    "Dual (mean)": 21.9,
    "Dual (attention 0)": 9.9,
    "Dual-VideoMix (attention 0)": 7.7,
    "Dual (attention 1)": 22.3,
    "Dual (attention 2)": 22.2,
    "ViLt (mean)": 18.6,
    "Vita (mean)": 7.9,
    "STT (mean)": 14.2,
}

# Format: [16-shot-val, 16-shot-test, 4-shot-val, 4-shot-test]
HMDB51_KSHOT = {
    "CoOp (attention 0)": [63.2, 67.6, 24.7, 21.7],
    "CoOp T (attention 1)": [69.0, 67.6, 27.8, 28.9],
    "Dual (attention 0)": [68.4, 60.6, 46.9, 43.5],
    "Dual T (attention 1)": [72.1, 61.0, 30.5, 28.7],
    "ViLT T": [42.6, 42.1, 23.2, 26.3],
    "Vita T": [62.6, 54.6, 29.4, 27.4],
}
