"""
Data Augmentation Module - Sanket-Svasthya
Functions for augmenting sign language training data.

Author: Team Sanket-Svasthya
Date: January 2026
"""

import numpy as np

# Fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)


def add_gaussian_noise(data: np.ndarray, factor: float = 0.02) -> np.ndarray:
    """
    Add Gaussian noise to data for augmentation.
    
    Args:
        data: Input array of any shape
        factor: Standard deviation of noise (default: 0.02)
        
    Returns:
        Augmented data with same shape as input
    """
    noise = np.random.normal(0, factor, data.shape)
    return data + noise


def augment_dataset(X: np.ndarray, y: np.ndarray, 
                    noise_factors: list = [0.01, 0.02]) -> tuple:
    """
    Augment dataset with multiple noise levels.
    
    Args:
        X: Input features of shape (N, seq_len, feature_dim)
        y: Labels of shape (N,)
        noise_factors: List of noise standard deviations
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    X_augmented = [X]
    y_augmented = [y]
    
    for factor in noise_factors:
        X_noisy = add_gaussian_noise(X, factor)
        X_augmented.append(X_noisy)
        y_augmented.append(y)
        print(f"Added augmentation with noise σ={factor}")
    
    X_final = np.concatenate(X_augmented, axis=0)
    y_final = np.concatenate(y_augmented, axis=0)
    
    print(f"Original: {len(X)} samples → Augmented: {len(X_final)} samples")
    
    return X_final, y_final


def temporal_shift(data: np.ndarray, max_shift: int = 3) -> np.ndarray:
    """
    Apply random temporal shift to sequence data.
    
    Args:
        data: Input sequence of shape (seq_len, feature_dim)
        max_shift: Maximum frames to shift
        
    Returns:
        Shifted sequence with same shape
    """
    shift = np.random.randint(-max_shift, max_shift + 1)
    
    if shift > 0:
        return np.concatenate([np.zeros((shift, data.shape[1])), data[:-shift]], axis=0)
    elif shift < 0:
        return np.concatenate([data[-shift:], np.zeros((-shift, data.shape[1]))], axis=0)
    else:
        return data


def scale_features(data: np.ndarray, scale_range: tuple = (0.9, 1.1)) -> np.ndarray:
    """
    Randomly scale feature values.
    
    Args:
        data: Input array
        scale_range: (min_scale, max_scale) tuple
        
    Returns:
        Scaled data
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale


if __name__ == "__main__":
    # Test augmentation
    print("Testing augmentation functions...")
    
    # Create dummy data
    X_dummy = np.random.randn(10, 30, 1662)
    y_dummy = np.arange(10)
    
    # Test augmentation
    X_aug, y_aug = augment_dataset(X_dummy, y_dummy, [0.01, 0.02])
    
    print(f"Input shape: {X_dummy.shape}")
    print(f"Output shape: {X_aug.shape}")
    print("Augmentation test passed!")
