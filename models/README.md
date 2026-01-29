# Model Architecture

## Overview

Sanket-Svasthya uses a dual-model architecture:

1. **Medical SignTransformer** - Dynamic signs (54 classes)
2. **ISL Static Classifier** - Alphabets (25 classes)

## Model Files

| File | Description |
|------|-------------|
| `model.py` | Model architecture definitions |
| `custom_layers.py` | Custom layers (if any) |

## Medical Sign Model

```
┌─────────────────────────────────────────┐
│         Input: (30, 1662)               │
├─────────────────────────────────────────┤
│  Dense(128, ReLU) + Dropout(0.2)        │
├─────────────────────────────────────────┤
│  LSTM(64, return_sequences=True)        │
├─────────────────────────────────────────┤
│  LSTM(64, return_sequences=False)       │
├─────────────────────────────────────────┤
│  Dense(64, ReLU) + Dropout(0.3)         │
├─────────────────────────────────────────┤
│  Dense(54, Softmax) → Output            │
└─────────────────────────────────────────┘
```

**Parameters:** ~270K

## ISL Alphabet Model

```
┌─────────────────────────────────────────┐
│         Input: (126,)                   │
├─────────────────────────────────────────┤
│  Dense(256, ReLU) + BatchNorm + Dropout │
├─────────────────────────────────────────┤
│  Dense(128, ReLU) + BatchNorm + Dropout │
├─────────────────────────────────────────┤
│  Dense(64, ReLU)                        │
├─────────────────────────────────────────┤
│  Dense(25, Softmax) → Output            │
└─────────────────────────────────────────┘
```

**Parameters:** ~50K

## Usage

```python
from models.model import get_medical_model, get_general_model, compile_model

# Create medical model
med_model = get_medical_model(input_shape=(30, 1662), num_classes=54)
med_model = compile_model(med_model, learning_rate=0.0001)

# Create ISL model
gen_model = get_general_model(input_shape=(126,), num_classes=25)
gen_model = compile_model(gen_model, learning_rate=0.001)
```
