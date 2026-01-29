# Inference Module

## Overview

Real-time and batch inference for sign language recognition.

## Files

| File | Description |
|------|-------------|
| `infer.py` | Single file inference |
| `realtime_demo.py` | Live webcam demo (Streamlit) |
| `tts.py` | Text-to-speech module |

## Quick Start

### Real-Time Demo
```bash
streamlit run realtime_demo.py
```

### Single File Inference
```bash
# Video (medical signs)
python infer.py --input video.mp4 --mode medical

# Image (alphabets)
python infer.py --input hand.jpg --mode general
```

## Python API

```python
from infer import SignRecognizer

# Medical signs
recognizer = SignRecognizer(mode='medical')
result = recognizer.predict('patient_sign.mp4')
print(f"Sign: {result['sign']}, Confidence: {result['confidence']:.2%}")

# Alphabets
recognizer = SignRecognizer(mode='general')
result = recognizer.predict('letter.jpg')
print(f"Letter: {result['sign']}")
```

## Text-to-Speech

```python
from tts import TTSEngine, speak

# Quick usage
speak("Hello patient")

# Advanced
tts = TTSEngine(rate=150, volume=0.9)
tts.speak("Doctor requested", blocking=False)
```

## Real-Time Features

- Dual mode: Medical / General
- Emergency alerts
- History tracking
- Staff panel (password: swastya)
