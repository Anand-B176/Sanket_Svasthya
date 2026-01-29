# User Interface Module

## Overview

Streamlit-based user interface for Sanket-Svasthya.

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |

## Running the UI

```bash
# From ui/ directory
streamlit run app.py

# Or from project root
streamlit run ui/app.py
```

## Features

- **Dual Mode:** Medical (54 signs) / General (25 letters)
- **Real-Time Video:** Live webcam feed with landmark visualization
- **Sign Detection:** Displays recognized sign with confidence
- **History Tracking:** Shows recent predictions
- **Emergency Alerts:** Flashing display for critical signs
- **Text-to-Speech:** Audio feedback for patients
- **Staff Panel:** Password-protected settings

## Configuration

Edit `app.py` to customize:

```python
SEQUENCE_LENGTH = 30      # Frames per prediction
THRESHOLD = 0.60          # Confidence threshold
PAUSE_DURATION = 1.5      # Seconds between predictions
STAFF_PASSWORD = "swastya"  # Staff panel password
```

## Screenshots

[Add screenshots here after running the demo]
