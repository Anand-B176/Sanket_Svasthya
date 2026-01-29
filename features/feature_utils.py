"""
Feature Extraction Utilities - Sanket-Svasthya
Common utility functions for feature processing.

Author: Team Sanket-Svasthya
Date: January 2026
"""

import numpy as np
import cv2
import mediapipe as mp


# Feature dimensions
POSE_DIM = 33 * 4      # 132
FACE_DIM = 468 * 3     # 1404
HAND_DIM = 21 * 3      # 63 per hand
TOTAL_DIM = POSE_DIM + FACE_DIM + 2 * HAND_DIM  # 1662


def extract_keypoints(results) -> np.ndarray:
    """
    Extract keypoints from MediaPipe Holistic results.
    
    Args:
        results: MediaPipe Holistic processing results
        
    Returns:
        Flattened numpy array of shape (1662,)
    """
    # Pose: 33 landmarks × 4 values (x, y, z, visibility)
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(POSE_DIM)
    
    # Face: 468 landmarks × 3 values (x, y, z)
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(FACE_DIM)
    
    # Hands: 21 landmarks × 3 values each
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(HAND_DIM)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(HAND_DIM)
    
    return np.concatenate([pose, face, lh, rh])


def extract_keypoints_normalized(results) -> np.ndarray:
    """
    Extract and normalize keypoints with nose-centered normalization.
    
    This provides translation invariance - same sign at different 
    screen positions produces the same normalized coordinates.
    
    Args:
        results: MediaPipe Holistic processing results
        
    Returns:
        Normalized flattened numpy array of shape (1662,)
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
    
    # Normalize: Shift all coordinates relative to nose position
    if results.pose_landmarks:
        ref_x = results.pose_landmarks.landmark[0].x  # Nose X
        ref_y = results.pose_landmarks.landmark[0].y  # Nose Y
        
        # Shift pose
        pose[:, 0] -= ref_x
        pose[:, 1] -= ref_y
        
        # Shift face
        if results.face_landmarks:
            face[:, 0] -= ref_x
            face[:, 1] -= ref_y
            
        # Shift hands
        if results.left_hand_landmarks:
            lh[:, 0] -= ref_x
            lh[:, 1] -= ref_y
            
        if results.right_hand_landmarks:
            rh[:, 0] -= ref_x
            rh[:, 1] -= ref_y
    
    return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), rh.flatten()])


def extract_hands_only(results) -> np.ndarray:
    """
    Extract only hand landmarks (for static alphabet recognition).
    
    Args:
        results: MediaPipe Holistic processing results
        
    Returns:
        Flattened numpy array of shape (126,) - both hands
    """
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(HAND_DIM)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(HAND_DIM)
    
    return np.concatenate([lh, rh])


def has_hands_detected(results) -> bool:
    """
    Check if any hands are detected in the frame.
    
    Args:
        results: MediaPipe Holistic processing results
        
    Returns:
        True if at least one hand is detected
    """
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None


def process_frame(frame: np.ndarray, holistic) -> tuple:
    """
    Process a single frame and extract features.
    
    Args:
        frame: BGR image from OpenCV
        holistic: MediaPipe Holistic instance
        
    Returns:
        Tuple of (results, keypoints, has_hands)
    """
    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = holistic.process(rgb)
    
    # Check for hands
    has_hands = has_hands_detected(results)
    
    # Extract keypoints
    keypoints = extract_keypoints_normalized(results) if has_hands else None
    
    return results, keypoints, has_hands
