# SANKET-SVASTHYA (Sign2Sound)
## Real-Time Sign Language Recognition for Healthcare Communication

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Competition](https://img.shields.io/badge/IEEE-Sign2Sound-red.svg)](https://ieee.org)

---

##  Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Team](#team)

---

##  Overview

**Sanket-Svasthya** is a real-time sign language recognition system specifically designed for healthcare environments. It bridges the communication gap between hearing-impaired patients and medical staff through:

- **54 Medical Signs** recognition (symptoms, conditions, emergencies)
- **25 ISL Alphabets** recognition (general communication)
- **Text-to-Speech** feedback for patient confirmation
- **Emergency Alert System** for critical signs (Heart attack, Stroke, etc.)
- **Smart Hospital Integration** with room/patient context

---

##  Features

| Feature | Description |
|---------|-------------|
| **Medical Mode** | 54 healthcare-specific signs |
| **General Mode** | 25 ISL alphabet signs |
| **Emergency Alerts** | Instant notification for critical signs |
| **Text-to-Speech** | Audio feedback for patients |
| **Staff Panel** | Password-protected settings |
| **Real-Time** | 15-20 FPS, <100ms latency |

---

## Installation

### Prerequisites
- Python 3.10+
- Webcam
- Windows/Linux/macOS

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/SIGN2SOUND_SanketSvasthya.git
cd SIGN2SOUND_SanketSvasthya
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Model Checkpoints
```bash
# Models are included in checkpoints/ folder
# Or download from Google Drive (if >100MB):
# https://drive.google.com/your-model-link
```

---

## Quick Start

### Run Real-Time Demo
```bash
cd inference
streamlit run realtime_demo.py
```

### Run Single Inference
```bash
python inference/infer.py --input video.mp4 --mode medical
```

### Train Model
```bash
python training/train.py --config training/config.yaml
```

---

##  Project Structure

```
SIGN2SOUND_SanketSvasthya/
│
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── LICENSE                      # MIT License
├── .gitignore                   # Git ignore rules
│
├── data/                        # Dataset information
│   ├── README.md                # Dataset sources & instructions
│   └── statistics.txt           # Dataset statistics
│
├── preprocessing/               # Data preprocessing
│   ├── preprocess.py            # Main preprocessing script
│   ├── augmentation.py          # Data augmentation
│   ├── extract_features.py      # Feature extraction
│   └── README.md                # Documentation
│
├── features/                    # Feature extraction
│   ├── hand_landmarks.py        # Hand detection
│   ├── pose_estimation.py       # Pose extraction
│   ├── facial_features.py       # Face features
│   ├── feature_utils.py         # Utilities
│   └── README.md                # Documentation
│
├── models/                      # Model architecture
│   ├── model.py                 # Main model
│   ├── custom_layers.py         # Custom layers
│   └── README.md                # Documentation
│
├── training/                    # Training pipeline
│   ├── train.py                 # Training script
│   ├── config.yaml              # Hyperparameters
│   ├── evaluate.py              # Evaluation
│   └── README.md                # Documentation
│
├── inference/                   # Inference & demo
│   ├── infer.py                 # Single inference
│   ├── realtime_demo.py         # Streamlit demo
│   ├── tts.py                   # Text-to-speech
│   └── README.md                # Documentation
│
├── ui/                          # User interface
│   ├── app.py                   # Streamlit app
│   └── README.md                # Documentation
│
├── results/                     # Performance results
│   ├── metrics.json             # All metrics
│   ├── confusion_matrix.png     # Confusion matrix
│   ├── training_curves.png      # Loss/accuracy curves
│   └── per_class_performance.csv # Per-class metrics
│
├── checkpoints/                 # Model weights
│   ├── medical_model.h5         # Medical model
│   ├── general_model.h5         # ISL model
│   └── README.md                # Download links
│
├── docs/                        # Documentation
│   ├── architecture_diagram.png # Architecture
│   ├── technical_report.md      # Full report
│   ├── performance_analysis.md  # Performance metrics
│   └── README.md                # Documentation index
│
└── tests/                       # Unit tests
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_inference.py
```

---

##  Usage

### Medical Mode (54 Signs)
```python
from inference.infer import SignRecognizer

recognizer = SignRecognizer(mode='medical')
result = recognizer.predict(video_path='patient_sign.mp4')
print(f"Detected: {result['sign']}, Confidence: {result['confidence']}")
```

### General Mode (ISL Alphabets)
```python
recognizer = SignRecognizer(mode='general')
result = recognizer.predict(video_path='alphabet_sign.mp4')
print(f"Letter: {result['sign']}")
```

### Real-Time with Webcam
```python
recognizer = SignRecognizer(mode='medical', use_webcam=True)
recognizer.run_realtime()
```

---

##  Training

### Prepare Data
```bash
python preprocessing/extract_features.py --input data/raw --output data/processed
```

### Train Medical Model
```bash
python training/train.py \
    --data data/processed \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001
```

### Evaluate Model
```bash
python training/evaluate.py --model checkpoints/best_model.h5
```

---

## Results

### Overall Performance
| Metric | Medical Model | General Model |
|--------|---------------|---------------|
| Accuracy | 85.2% | 91.4% |
| Precision | 84.7% | 90.8% |
| Recall | 83.9% | 91.1% |
| F1-Score | 84.3% | 90.9% |

### Inference Speed
| Metric | Value |
|--------|-------|
| Latency | 60.7 ms |
| FPS | 16.5 |
| Real-Time |  Yes |

### Top Performing Signs
| Sign | F1-Score |
|------|----------|
| Heart attack | 94.2% |
| Headache | 93.1% |
| Vomit | 92.8% |
| Brain stroke | 91.5% |

---

## 🔧 Configuration

### config.yaml
```yaml
model:
  sequence_length: 30
  num_classes: 54
  feature_dim: 1662

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  early_stopping_patience: 10

augmentation:
  noise_factor: [0.01, 0.02]
  augmentation_multiplier: 3
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_model.py -v
```

---

##  Documentation

- [Technical Report](docs/technical_report.md)
- [Performance Analysis](docs/performance_analysis.md)
- [Architecture Diagram](docs/architecture_diagram.png)
- [Dataset Documentation](data/README.md)
- [Training Guide](training/README.md)
- [Inference Guide](inference/README.md)

---

## Reproducibility

### Fixed Random Seeds
```python
import numpy as np
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

### Environment
```bash
# Export environment
pip freeze > requirements.txt

# Reproduce
pip install -r requirements.txt
```

---

##  Team

**Team Sanket-Svasthya**

| Role | Member |
|------|--------|
| Lead Developer | Anand Babu |
| ML Engineer | Edwin K Jayesh|
| UI/UX Designer | Jacob Rajeev |

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- IEEE Sign2Sound Competition
- MediaPipe Team (Google)
- TensorFlow Team
- Dataset Providers


---

*Developed for IEEE Sign2Sound Competition 2026*
