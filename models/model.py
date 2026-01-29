"""
Model Architecture - Sanket-Svasthya
Neural network models for sign language recognition.

Author: Team Sanket-Svasthya
Date: January 2026
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_medical_model(input_shape: tuple = (30, 1662), 
                      num_classes: int = 54) -> keras.Model:
    """
    Create the medical sign recognition model.
    
    Architecture: LSTM-based temporal model for dynamic sign recognition.
    
    Args:
        input_shape: (sequence_length, feature_dim) - default (30, 1662)
        num_classes: Number of output classes - default 54
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Feature compression
        layers.Dense(128, activation='relu', name='feature_compress'),
        layers.Dropout(0.2),
        
        # Temporal processing with LSTM
        layers.LSTM(64, return_sequences=True, name='lstm_1'),
        layers.LSTM(64, return_sequences=False, name='lstm_2'),
        
        # Classification head
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='MedicalSignModel')
    
    return model


def get_general_model(input_shape: tuple = (126,), 
                      num_classes: int = 25) -> keras.Model:
    """
    Create the ISL alphabet recognition model.
    
    Architecture: Dense network for static pose classification.
    
    Args:
        input_shape: (feature_dim,) - default (126,) for hands only
        num_classes: Number of output classes - default 25 (A-Y)
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Dense layers with batch normalization
        layers.Dense(256, activation='relu', name='dense_1'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', name='dense_3'),
        
        # Output
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='GeneralSignModel')
    
    return model


def get_transformer_model(input_shape: tuple = (30, 1662), 
                          num_classes: int = 54,
                          num_heads: int = 4,
                          ff_dim: int = 128) -> keras.Model:
    """
    Create a Transformer-based model for sign recognition.
    
    Note: This is a more complex alternative to the LSTM model.
    Use when more computational resources are available.
    
    Args:
        input_shape: (sequence_length, feature_dim)
        num_classes: Number of output classes
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Feature compression
    x = layers.Dense(128)(inputs)
    
    # Positional encoding (simplified)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_encoding = layers.Embedding(input_shape[0], 128)(positions)
    x = x + pos_encoding
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=128 // num_heads
    )(x, x)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # Feed-forward network
    ff_output = layers.Dense(ff_dim, activation='relu')(x)
    ff_output = layers.Dense(128)(ff_output)
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization()(x)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='TransformerSignModel')
    
    return model


def compile_model(model: keras.Model, 
                  learning_rate: float = 0.0001) -> keras.Model:
    """
    Compile model with standard configuration.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing model architectures...")
    
    # Medical model
    med_model = get_medical_model()
    med_model = compile_model(med_model)
    print(f"\nMedical Model:")
    med_model.summary()
    
    # General model
    gen_model = get_general_model()
    gen_model = compile_model(gen_model, learning_rate=0.001)
    print(f"\nGeneral Model:")
    gen_model.summary()
    
    print("\nModel tests passed!")
