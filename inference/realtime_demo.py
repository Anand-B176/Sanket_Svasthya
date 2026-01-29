"""
Real-Time Demo - Sanket-Svasthya
Streamlit application for live sign language recognition.

Run: streamlit run realtime_demo.py

Author: Team Sanket-Svasthya
Date: January 2026
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import sys
import tensorflow as tf
import pyttsx3
import threading
from datetime import datetime
import base64

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_utils import extract_keypoints_normalized, has_hands_detected

# ====================== CONFIGURATION ======================
SEQUENCE_LENGTH = 30
THRESHOLD = 0.60
PAUSE_DURATION = 1.5
STAFF_PASSWORD = "swastya"

SHORTCUTS = {'D': 'DOCTOR', 'N': 'NURSE', 'H': 'HELP'}
EMERGENCY_SIGNS = ["Heart attack", "Cerebral hemorrhage", "Brain stroke", 
                   "Poisoning", "Vomit", "Fever"]

# Sign name mapping
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

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_models():
    """Load trained models with caching."""
    med, gen = None, None
    checkpoints_dir = "../checkpoints"
    
    med_path = os.path.join(checkpoints_dir, "medical_model.h5")
    gen_path = os.path.join(checkpoints_dir, "general_model.h5")
    
    if os.path.exists(med_path):
        med = tf.keras.models.load_model(med_path)
        print(f"✅ Loaded medical model")
    
    if os.path.exists(gen_path):
        gen = tf.keras.models.load_model(gen_path)
        print(f"✅ Loaded general model")
    
    return med, gen


@st.cache_resource
def load_classes():
    """Load class labels."""
    classes_path = "../checkpoints/classes.npy"
    if os.path.exists(classes_path):
        return np.load(classes_path, allow_pickle=True)
    return None


# ====================== TTS ======================
def speak(text: str):
    """Non-blocking text-to-speech."""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=_speak, daemon=True).start()


# ====================== ALERTS ======================
def send_alert(alert_type: str, message: str, room: str, patient: str):
    """Send alert to console (extensible to SMS/push)."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print("\n" + "="*60)
    print(f"🚨 ALERT - {alert_type} 🚨")
    print(f"   Time: {timestamp}")
    print(f"   Room: {room}")
    print(f"   Patient: {patient}")
    print(f"   Message: {message}")
    print("="*60 + "\n")


# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="Sanket-Svasthya", page_icon="🏥", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #0d1117 50%, #0a0a0a 100%);
        color: white;
    }
    .result-text {
        font-size: 4rem;
        font-weight: 800;
        color: #00ff88;
        text-align: center;
        text-shadow: 0 0 40px rgba(0,255,136,0.6);
        margin: 30px 0;
    }
    .brand-name {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00ff88, #00f5ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ====================== STATE ======================
if 'mode' not in st.session_state: st.session_state.mode = 'medical'
if 'history' not in st.session_state: st.session_state.history = []
if 'last_pred_time' not in st.session_state: st.session_state.last_pred_time = 0
if 'last_pred_sign' not in st.session_state: st.session_state.last_pred_sign = ""
if 'room_number' not in st.session_state: st.session_state.room_number = "305"
if 'patient_name' not in st.session_state: st.session_state.patient_name = "Patient"
if 'doctor_name' not in st.session_state: st.session_state.doctor_name = "Dr. Sharma"

# ====================== HEADER ======================
st.markdown("<div class='brand-name'>🏥 SANKET-SVASTHYA</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>Real-Time Sign Language Recognition for Healthcare</p>", unsafe_allow_html=True)

# ====================== LOAD MODELS ======================
med_model, gen_model = load_models()
classes = load_classes()
gen_classes = [chr(i) for i in range(65, 90)]

# ====================== MODE SWITCH ======================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    mode_cols = st.columns(2)
    with mode_cols[0]:
        if st.button("🏥 MEDICAL", use_container_width=True):
            st.session_state.mode = 'medical'
    with mode_cols[1]:
        if st.button("🗣️ GENERAL", use_container_width=True):
            st.session_state.mode = 'general'

st.markdown(f"**Current Mode:** {st.session_state.mode.upper()}")

# ====================== MAIN INTERFACE ======================
col_vid, col_result = st.columns([2, 1])

with col_vid:
    frame_window = st.image([])

with col_result:
    result_placeholder = st.empty()
    history_placeholder = st.empty()
    debug_placeholder = st.empty()

# ====================== MAIN LOOP ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
sequence = []

with mp_holistic.Holistic(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        # Draw landmarks
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Check for hands
        has_hands = has_hands_detected(results)
        
        if not has_hands:
            sequence = []
            debug_placeholder.text("👋 Show your hands...")
            frame_window.image(frame, channels="BGR")
            continue
        
        # Extract features
        kp = extract_keypoints_normalized(results)
        sequence.append(kp)
        sequence = sequence[-SEQUENCE_LENGTH:]
        
        debug_placeholder.text(f"Buffer: {len(sequence)}/{SEQUENCE_LENGTH}")
        
        now = time.time()
        if len(sequence) == SEQUENCE_LENGTH and (now - st.session_state.last_pred_time > PAUSE_DURATION):
            
            if st.session_state.mode == 'medical' and med_model and classes is not None:
                # Medical prediction
                res = med_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                idx = np.argmax(res)
                conf = res[idx]
                
                if conf > THRESHOLD:
                    sign_id = classes[idx] if idx < len(classes) else f"Sign_{idx+1}"
                    sign_name = SIGN_NAMES.get(sign_id, sign_id)
                    
                    st.session_state.last_pred_sign = sign_name
                    st.session_state.history.append(sign_name)
                    st.session_state.last_pred_time = now
                    
                    result_placeholder.markdown(f"<div class='result-text'>{sign_name}</div>", unsafe_allow_html=True)
                    history_placeholder.text(f"History: {' → '.join(st.session_state.history[-5:])}")
                    
                    if sign_name in EMERGENCY_SIGNS:
                        send_alert("EMERGENCY", sign_name, st.session_state.room_number, st.session_state.patient_name)
                    
                    speak(sign_name)
            
            elif st.session_state.mode == 'general' and gen_model:
                # General (alphabet) prediction
                raw_hands = np.concatenate([
                    np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63),
                    np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
                ])
                
                res = gen_model.predict(np.expand_dims(raw_hands, axis=0), verbose=0)[0]
                idx = np.argmax(res)
                conf = res[idx]
                
                if conf > 0.55:
                    letter = gen_classes[idx]
                    
                    st.session_state.last_pred_sign = letter
                    st.session_state.history.append(letter)
                    st.session_state.last_pred_time = now
                    
                    result_placeholder.markdown(f"<div class='result-text'>{letter}</div>", unsafe_allow_html=True)
                    history_placeholder.text(f"Spelled: {''.join(st.session_state.history[-10:])}")
                    
                    speak(letter)
        
        frame_window.image(frame, channels="BGR")

cap.release()
