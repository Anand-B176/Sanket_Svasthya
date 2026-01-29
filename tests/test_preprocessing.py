"""
Unit Tests - Preprocessing
Sanket-Svasthya Sign Language Recognition

Author: Team Sanket-Svasthya
Date: January 2026
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_augmentation():
    """Test data augmentation functions."""
    from preprocessing.augmentation import add_gaussian_noise, augment_dataset
    
    # Test single noise addition
    data = np.ones((10, 30, 1662))
    noisy = add_gaussian_noise(data, factor=0.01)
    
    assert noisy.shape == data.shape
    assert not np.array_equal(noisy, data)  # Should be different
    print("✅ Gaussian noise test passed")
    
    # Test dataset augmentation
    X = np.random.randn(100, 30, 1662)
    y = np.arange(100)
    
    X_aug, y_aug = augment_dataset(X, y, noise_factors=[0.01, 0.02])
    
    assert len(X_aug) == 300  # 3x original
    assert len(y_aug) == 300
    print("✅ Dataset augmentation test passed")


def test_pad_or_truncate():
    """Test sequence padding and truncation."""
    from preprocessing.preprocess import pad_or_truncate
    
    # Test truncation
    long_seq = np.ones((50, 1662))
    result = pad_or_truncate(long_seq, 30)
    assert result.shape == (30, 1662)
    print("✅ Truncation test passed")
    
    # Test padding
    short_seq = np.ones((10, 1662))
    result = pad_or_truncate(short_seq, 30)
    assert result.shape == (30, 1662)
    assert np.sum(result[10:]) == 0  # Padded with zeros
    print("✅ Padding test passed")


if __name__ == "__main__":
    print("Running preprocessing tests...\n")
    
    test_augmentation()
    test_pad_or_truncate()
    
    print("\n✅ All preprocessing tests passed!")
