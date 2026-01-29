"""
Inference Script - Sanket-Svasthya
Single file/video inference for sign language recognition.

Usage:
    python infer.py --input video.mp4 --mode medical
    python infer.py --input image.jpg --mode general

Author: Team Sanket-Svasthya
Date: January 2026
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import argparse
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_utils import extract_keypoints_normalized, extract_hands_only, has_hands_detected

# Configuration
SEQUENCE_LENGTH = 30
THRESHOLD = 0.60

# Sign names
SIGN_NAMES = {
    "Sign_01": "Depression", "Sign_02": "Appendicitis", "Sign_03": "Audiogram",
    "Sign_04": "Audiometry", "Sign_05": "Future", "Sign_06": "Medical scheduling",
    "Sign_07": "Chronic disease", "Sign_08": "Toothache", "Sign_09": "To have",
    "Sign_10": "Diabetes", "Sign_11": "Yesterday", "Sign_12": "Today",
    "Sign_13": "Asthma spray", "Sign_14": "Vomit", "Sign_15": "Anxiety",
    "Sign_16": "Tomorrow", "Sign_17": "Blood pressure", "Sign_18": "Chest",
    "Sign_19": "Past", "Sign_20": "Now", "Sign_21": "Heart attack",
    "Sign_22": "Surgery", "Sign_23": "Psychosis", "Sign_24": "Hearing impaired",
    "Sign_25": "Cerebral hemorrhage", "Sign_26": "Sicken", "Sign_27": "Alcohol poisoning",
    "Sign_28": "Poisoning", "Sign_29": "Stomach", "Sign_30": "To can",
    "Sign_31": "Medical consultation", "Sign_32": "Deaf", "Sign_33": "To have not",
    "Sign_34": "To cannot", "Sign_35": "Semi-hearing", "Sign_36": "Medical prescription",
    "Sign_37": "Respiratory infection", "Sign_38": "Mumps", "Sign_39": "Hand tendon rupture",
    "Sign_40": "Stable", "Sign_41": "Lungs", "Sign_42": "Year",
    "Sign_43": "Kidney stone", "Sign_44": "Headache", "Sign_45": "A cold",
    "Sign_46": "Medical records", "Sign_47": "Brain stroke", "Sign_48": "Age",
    "Sign_49": "Cough", "Sign_50": "Needle", "Sign_51": "Fever",
    "Sign_52": "Injection", "Sign_53": "Hand Prickle", "Sign_54": "Vaccinate",
    "Sign_55": "Tonsillitis"
}


class SignRecognizer:
    """
    Sign Language Recognition inference class.
    
    Supports both medical signs (video sequences) and 
    general ISL alphabets (single frames).
    """
    
    def __init__(self, mode: str = 'medical', 
                 model_path: str = None,
                 classes_path: str = None):
        """
        Initialize recognizer.
        
        Args:
            mode: 'medical' or 'general'
            model_path: Path to model weights
            classes_path: Path to class labels
        """
        self.mode = mode
        self.sequence_length = SEQUENCE_LENGTH
        self.threshold = THRESHOLD
        
        # Default paths
        checkpoints_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
        
        if model_path is None:
            if mode == 'medical':
                model_path = os.path.join(checkpoints_dir, 'medical_model.h5')
            else:
                model_path = os.path.join(checkpoints_dir, 'general_model.h5')
        
        if classes_path is None:
            classes_path = os.path.join(checkpoints_dir, 'classes.npy')
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Loaded model: {model_path}")
        
        # Load classes
        if os.path.exists(classes_path):
            self.classes = np.load(classes_path, allow_pickle=True)
        else:
            self.classes = None
        
        # ISL alphabet classes
        self.alphabet_classes = [chr(i) for i in range(65, 90)]
        
        # Initialize MediaPipe
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
    
    def predict_video(self, video_path: str) -> dict:
        """
        Predict sign from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with prediction results
        """
        cap = cv2.VideoCapture(video_path)
        sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)
            
            if has_hands_detected(results):
                kp = extract_keypoints_normalized(results)
                sequence.append(kp)
        
        cap.release()
        
        if len(sequence) < self.sequence_length:
            # Pad with zeros
            padding = [np.zeros(1662)] * (self.sequence_length - len(sequence))
            sequence = sequence + padding
        else:
            sequence = sequence[:self.sequence_length]
        
        # Predict
        input_data = np.expand_dims(sequence, axis=0)
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        idx = np.argmax(predictions)
        confidence = predictions[idx]
        
        if self.classes is not None:
            sign_id = self.classes[idx]
            sign_name = SIGN_NAMES.get(sign_id, sign_id)
        else:
            sign_name = f"Sign_{idx+1}"
        
        return {
            'sign': sign_name,
            'confidence': float(confidence),
            'sign_id': sign_id if self.classes is not None else idx,
            'all_predictions': {
                SIGN_NAMES.get(self.classes[i] if self.classes else f"Sign_{i}", f"Sign_{i}"): float(predictions[i])
                for i in np.argsort(predictions)[-5:][::-1]
            }
        }
    
    def predict_image(self, image_path: str) -> dict:
        """
        Predict alphabet from single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with prediction results
        """
        frame = cv2.imread(image_path)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)
        
        if not has_hands_detected(results):
            return {'error': 'No hands detected'}
        
        # Extract hand features only
        hands = extract_hands_only(results)
        
        # Predict
        input_data = np.expand_dims(hands, axis=0)
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        idx = np.argmax(predictions)
        confidence = predictions[idx]
        letter = self.alphabet_classes[idx]
        
        return {
            'sign': letter,
            'confidence': float(confidence),
            'top_5': {
                self.alphabet_classes[i]: float(predictions[i])
                for i in np.argsort(predictions)[-5:][::-1]
            }
        }
    
    def predict(self, input_path: str) -> dict:
        """
        Predict sign from input file (video or image).
        
        Args:
            input_path: Path to input file
            
        Returns:
            Dictionary with prediction results
        """
        if self.mode == 'medical':
            return self.predict_video(input_path)
        else:
            return self.predict_image(input_path)


def main():
    parser = argparse.ArgumentParser(description="Sign language inference")
    parser.add_argument("--input", type=str, required=True, help="Input file path")
    parser.add_argument("--mode", type=str, default="medical", choices=["medical", "general"])
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    # Create recognizer
    recognizer = SignRecognizer(mode=args.mode)
    
    # Predict
    result = recognizer.predict(args.input)
    
    # Print result
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Sign: {result.get('sign', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0):.2%}")
    print("="*50)
    
    # Save to JSON if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return result


if __name__ == "__main__":
    main()
