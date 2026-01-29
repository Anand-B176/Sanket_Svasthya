"""
Training Script - Sanket-Svasthya
Main training pipeline for sign language models.

Author: Team Sanket-Svasthya
Date: January 2026
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import yaml
import argparse
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import get_medical_model, get_general_model, compile_model
from preprocessing.augmentation import augment_dataset

# Fixed random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_path: str) -> tuple:
    """
    Load preprocessed data from numpy files.
    
    Args:
        data_path: Path to directory containing .npy files
        
    Returns:
        Tuple of (X, y, classes)
    """
    X = np.load(os.path.join(data_path, 'X.npy'))
    y = np.load(os.path.join(data_path, 'y.npy'))
    classes = np.load(os.path.join(data_path, 'classes.npy'), allow_pickle=True)
    
    print(f"Loaded data: X={X.shape}, y={y.shape}, classes={len(classes)}")
    
    return X, y, classes


def create_callbacks(checkpoint_path: str, patience: int = 10) -> list:
    """
    Create training callbacks.
    
    Args:
        checkpoint_path: Path to save best model
        patience: Early stopping patience
        
    Returns:
        List of callbacks
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    return callbacks


def plot_training_history(history, save_path: str):
    """
    Plot and save training curves.
    
    Args:
        history: Keras training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', color='cyan')
    axes[0].plot(history.history['val_loss'], label='Validation Loss', color='magenta')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', color='cyan')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='magenta')
    axes[1].set_title('Training vs Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def train(config: dict):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    # Load data
    X, y, classes = load_data(config['data']['path'])
    num_classes = len(classes)
    
    # Augment data
    if config['augmentation']['enabled']:
        X, y = augment_dataset(
            X, y, 
            noise_factors=config['augmentation']['noise_factors']
        )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config['data']['test_size'],
        random_state=SEED,
        stratify=y
    )
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}")
    
    # Create model
    if config['model']['type'] == 'medical':
        model = get_medical_model(
            input_shape=(config['model']['sequence_length'], config['model']['feature_dim']),
            num_classes=num_classes
        )
    else:
        model = get_general_model(
            input_shape=(config['model']['feature_dim'],),
            num_classes=num_classes
        )
    
    # Compile
    model = compile_model(model, learning_rate=config['training']['learning_rate'])
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(
        checkpoint_path=config['training']['checkpoint_path'],
        patience=config['training']['early_stopping_patience']
    )
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(config['training']['final_model_path'])
    print(f"Model saved to {config['training']['final_model_path']}")
    
    # Save classes
    np.save(config['training']['classes_path'], classes)
    
    # Plot training curves
    plot_training_history(history, config['training']['curves_path'])
    
    # Print final results
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {max(history.history['val_accuracy']):.2%}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sign language model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Train
    train(config)
