"""
Unit Tests - Model Testing
Sanket-Svasthya Sign Language Recognition

Author: Team Sanket-Svasthya
Date: January 2026
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_medical_model_creation():
    """Test medical model can be created."""
    from models.model import get_medical_model, compile_model
    
    model = get_medical_model(input_shape=(30, 1662), num_classes=54)
    model = compile_model(model)
    
    assert model is not None
    assert model.input_shape == (None, 30, 1662)
    assert model.output_shape == (None, 54)
    print("✅ Medical model creation test passed")


def test_general_model_creation():
    """Test general model can be created."""
    from models.model import get_general_model, compile_model
    
    model = get_general_model(input_shape=(126,), num_classes=25)
    model = compile_model(model)
    
    assert model is not None
    assert model.input_shape == (None, 126)
    assert model.output_shape == (None, 25)
    print("✅ General model creation test passed")


def test_medical_model_inference():
    """Test medical model can perform inference."""
    from models.model import get_medical_model, compile_model
    
    model = get_medical_model()
    model = compile_model(model)
    
    # Create dummy input
    dummy_input = np.random.randn(1, 30, 1662).astype(np.float32)
    
    # Predict
    output = model.predict(dummy_input, verbose=0)
    
    assert output.shape == (1, 54)
    assert np.allclose(output.sum(), 1.0, atol=0.01)  # Softmax sums to 1
    print("✅ Medical model inference test passed")


def test_general_model_inference():
    """Test general model can perform inference."""
    from models.model import get_general_model, compile_model
    
    model = get_general_model()
    model = compile_model(model)
    
    # Create dummy input
    dummy_input = np.random.randn(1, 126).astype(np.float32)
    
    # Predict
    output = model.predict(dummy_input, verbose=0)
    
    assert output.shape == (1, 25)
    assert np.allclose(output.sum(), 1.0, atol=0.01)
    print("✅ General model inference test passed")


if __name__ == "__main__":
    print("Running model tests...\n")
    
    test_medical_model_creation()
    test_general_model_creation()
    test_medical_model_inference()
    test_general_model_inference()
    
    print("\n✅ All model tests passed!")
