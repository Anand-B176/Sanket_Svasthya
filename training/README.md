# Training Module

## Overview

Complete training pipeline for sign language models.

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `evaluate.py` | Model evaluation |
| `config.yaml` | Hyperparameters |

## Quick Start

```bash
python train.py --config config.yaml
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  type: "medical"
  sequence_length: 30
  feature_dim: 1662

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  early_stopping_patience: 10

augmentation:
  enabled: true
  noise_factors: [0.01, 0.02]
```

## Training Process

1. Load preprocessed data
2. Apply data augmentation (3× dataset)
3. Train/validation split (80/20)
4. Train with early stopping
5. Save best model checkpoint
6. Plot training curves

## Callbacks

- **EarlyStopping:** Patience 10, restore best weights
- **ReduceLROnPlateau:** Factor 0.5, patience 5
- **ModelCheckpoint:** Save best by validation accuracy

## Evaluation

```bash
python evaluate.py --model ../checkpoints/best_model.h5 --data ../data/processed
```

Generates:
- Classification report
- Confusion matrix
- Per-class metrics
