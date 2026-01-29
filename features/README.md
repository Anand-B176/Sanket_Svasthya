# Feature Extraction Module

## Overview

This module handles feature extraction from video frames using MediaPipe Holistic.

## Files

| File | Description |
|------|-------------|
| `feature_utils.py` | Core extraction functions |
| `hand_landmarks.py` | Hand-specific processing |
| `pose_estimation.py` | Pose detection |
| `facial_features.py` | Face mesh extraction |

## Key Functions

### `extract_keypoints(results)`
Extract raw keypoints from MediaPipe results.

### `extract_keypoints_normalized(results)`
Extract keypoints with nose-centered normalization (translation invariance).

### `extract_hands_only(results)`
Extract only hand landmarks for static alphabet recognition.

## Usage

```python
import cv2
import mediapipe as mp
from feature_utils import extract_keypoints_normalized, has_hands_detected

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Process frame
frame = cv2.imread('frame.jpg')
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = holistic.process(rgb)

# Check for hands
if has_hands_detected(results):
    features = extract_keypoints_normalized(results)  # (1662,)
```

## Normalization

Nose-centered normalization provides translation invariance:
- All coordinates shifted relative to nose position
- Same sign at different positions → Same features
