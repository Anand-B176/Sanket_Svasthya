"""
Preprocessing Module - Sanket-Svasthya
Main preprocessing script for sign language data.

This script processes raw video frames into feature vectors
suitable for model training and inference.

Author: Team Sanket-Svasthya
Date: January 2026
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import argparse

# Fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Configuration
SEQUENCE_LENGTH = 30
FEATURE_DIM = 1662


def process_video_folder(folder_path: str, holistic) -> np.ndarray:
    """
    Process a folder of video frames and extract features.
    
    Args:
        folder_path: Path to folder containing frame images
        holistic: MediaPipe Holistic instance
        
    Returns:
        numpy array of shape (num_frames, 1662)
    """
    frame_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    features = []
    
    for frame_file in frame_files:
        frame_path = os.path.join(folder_path, frame_file)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            continue
            
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = holistic.process(rgb)
        
        # Extract keypoints
        kp = extract_keypoints_normalized(results)
        features.append(kp)
    
    return np.array(features)


def extract_keypoints_normalized(results) -> np.ndarray:
    """
    Extract and normalize keypoints from MediaPipe results.
    
    Applies nose-centered normalization for translation invariance.
    
    Args:
        results: MediaPipe Holistic results
        
    Returns:
        Flattened numpy array of shape (1662,)
    """
    # Extract raw landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    
    # Nose-centered normalization
    if results.pose_landmarks:
        ref_x = results.pose_landmarks.landmark[0].x
        ref_y = results.pose_landmarks.landmark[0].y
        
        pose[:, 0] -= ref_x
        pose[:, 1] -= ref_y
        
        if results.face_landmarks:
            face[:, 0] -= ref_x
            face[:, 1] -= ref_y
            
        if results.left_hand_landmarks:
            lh[:, 0] -= ref_x
            lh[:, 1] -= ref_y
            
        if results.right_hand_landmarks:
            rh[:, 0] -= ref_x
            rh[:, 1] -= ref_y
    
    return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), rh.flatten()])


def pad_or_truncate(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate sequence to target length.
    
    Args:
        sequence: Input sequence of shape (N, 1662)
        target_length: Desired sequence length
        
    Returns:
        Sequence of shape (target_length, 1662)
    """
    if len(sequence) >= target_length:
        return sequence[:target_length]
    else:
        padding = np.zeros((target_length - len(sequence), FEATURE_DIM))
        return np.concatenate([sequence, padding], axis=0)


def preprocess_dataset(input_dir: str, output_dir: str, sequence_length: int = 30):
    """
    Preprocess entire dataset from raw frames to numpy arrays.
    
    Args:
        input_dir: Path to raw dataset
        output_dir: Path to save processed data
        sequence_length: Fixed sequence length for all samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    mp_holistic = mp.solutions.holistic
    
    X_all = []
    y_all = []
    classes = []
    
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:
        
        # Iterate through sign folders
        for sign_folder in tqdm(sorted(os.listdir(input_dir)), desc="Processing signs"):
            sign_path = os.path.join(input_dir, sign_folder)
            
            if not os.path.isdir(sign_path):
                continue
                
            sign_id = sign_folder.split('_')[0] + '_' + sign_folder.split('_')[1]
            
            if sign_id not in classes:
                classes.append(sign_id)
            
            label = classes.index(sign_id)
            
            # Find color frames folder
            color_frames_path = os.path.join(sign_path, '02 Color Frames')
            
            if os.path.exists(color_frames_path):
                features = process_video_folder(color_frames_path, holistic)
                
                if len(features) > 0:
                    features = pad_or_truncate(features, sequence_length)
                    X_all.append(features)
                    y_all.append(label)
    
    # Convert to numpy arrays
    X = np.array(X_all)
    y = np.array(y_all)
    
    # Save
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    np.save(os.path.join(output_dir, 'classes.npy'), np.array(classes))
    
    print(f"Preprocessing complete!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Classes: {len(classes)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess sign language dataset")
    parser.add_argument("--input", type=str, required=True, help="Input directory with raw data")
    parser.add_argument("--output", type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--seq_length", type=int, default=30, help="Sequence length")
    
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.output, args.seq_length)
