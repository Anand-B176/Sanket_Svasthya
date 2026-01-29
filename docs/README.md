# Documentation

## Overview

Complete documentation for Sanket-Svasthya Sign Language Recognition System.

## Files

| File | Description |
|------|-------------|
| `technical_report.pdf` | Complete technical report |
| `architecture_diagram.png` | Model architecture visualization |
| `system_pipeline.png` | End-to-end pipeline diagram |
| `dataset_preprocessing.md` | Data preprocessing details |
| `training_details.md` | Training procedure documentation |

## Technical Report Contents

1. Executive Summary
2. System Architecture
3. Dataset Description
4. Preprocessing Pipeline
5. Model Architecture
6. Training Procedure
7. Evaluation Results
8. Performance Metrics
9. Challenges & Solutions
10. Future Work

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SANKET-SVASTHYA SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WEBCAM → MediaPipe Holistic → Feature Extraction (1662-dim)   │
│                        ↓                                        │
│         ┌───────── Mode Switch ─────────┐                       │
│         ↓                               ↓                       │
│   Medical Model                   General Model                 │
│   (LSTM, 54 classes)             (Dense, 25 classes)           │
│         ↓                               ↓                       │
│   Sign Prediction                Letter Prediction              │
│         ↓                               ↓                       │
│         └───────── Output ──────────────┘                       │
│                        ↓                                        │
│   Display + TTS + Emergency Alert (if applicable)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Links

- [Main README](../README.md)
- [Training Guide](../training/README.md)
- [Inference Guide](../inference/README.md)
- [Dataset Info](../data/README.md)
