"""
Unit Tests - Inference
Sanket-Svasthya Sign Language Recognition

Author: Team Sanket-Svasthya
Date: January 2026
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_feature_extraction():
    """Test feature extraction dimensions."""
    from features.feature_utils import POSE_DIM, FACE_DIM, HAND_DIM, TOTAL_DIM
    
    assert POSE_DIM == 132    # 33 × 4
    assert FACE_DIM == 1404   # 468 × 3
    assert HAND_DIM == 63     # 21 × 3
    assert TOTAL_DIM == 1662  # Sum of all
    
    print("✅ Feature dimension test passed")


def test_tts_engine():
    """Test TTS engine initialization."""
    from inference.tts import TTSEngine
    
    tts = TTSEngine(rate=150, volume=0.9)
    
    assert tts.rate == 150
    assert tts.volume == 0.9
    
    # List voices should work
    voices = tts.list_voices()
    assert isinstance(voices, list)
    
    print("✅ TTS engine test passed")


def test_sign_names_mapping():
    """Test sign name mappings are complete."""
    from inference.infer import SIGN_NAMES
    
    # Should have 55 entries (Sign_01 to Sign_55)
    assert len(SIGN_NAMES) == 55
    
    # Check a few known mappings
    assert SIGN_NAMES["Sign_01"] == "Depression"
    assert SIGN_NAMES["Sign_21"] == "Heart attack"
    assert SIGN_NAMES["Sign_55"] == "Tonsillitis"
    
    print("✅ Sign names mapping test passed")


if __name__ == "__main__":
    print("Running inference tests...\n")
    
    test_feature_extraction()
    test_tts_engine()
    test_sign_names_mapping()
    
    print("\n✅ All inference tests passed!")
