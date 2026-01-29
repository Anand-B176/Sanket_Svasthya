# Preprocessing Module

## Overview

This module handles data preprocessing for sign language recognition:
- Video frame extraction
- MediaPipe feature extraction
- Nose-centered normalization
- Sequence padding/truncation
- Data augmentation

## Files

| File | Description |
|------|-------------|
| `preprocess.py` | Main preprocessing script |
| `augmentation.py` | Data augmentation functions |
| `extract_features.py` | Feature extraction from frames |

## Usage

### Process Raw Dataset

```bash
python preprocess.py --input ../../raw_data --output ../data/processed --seq_length 30
```

### Apply Augmentation

```python
from augmentation import augment_dataset

X_aug, y_aug = augment_dataset(X, y, noise_factors=[0.01, 0.02])
```

## Pipeline

```
Raw Video Frames (JPG)
      ↓
MediaPipe Holistic Processing
      ↓
1662-dim Feature Vector per Frame
      ↓
Nose-Centered Normalization
      ↓
Pad/Truncate to 30 Frames
      ↓
Save as NumPy Arrays (.npy)
```

## Feature Dimensions

| Component | Landmarks | Features |
|-----------|-----------|----------|
| Pose | 33 × 4 | 132 |
| Face | 468 × 3 | 1,404 |
| Left Hand | 21 × 3 | 63 |
| Right Hand | 21 × 3 | 63 |
| **Total** | - | **1,662** |
