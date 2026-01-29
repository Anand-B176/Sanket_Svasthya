# Model Checkpoints

## Download Links

Due to GitHub file size limits, model checkpoints are hosted externally.

### Medical Sign Model
- **File:** `medical_model.h5`
- **Size:** ~2 MB
- **Classes:** 54 medical signs
- **Architecture:** LSTM-based temporal model
- **Download:** [Google Drive Link - Add after upload]

### General ISL Model
- **File:** `general_model.h5`
- **Size:** ~500 KB
- **Classes:** 25 ISL alphabets (A-Y)
- **Architecture:** Dense neural network
- **Download:** [Google Drive Link - Add after upload]

### Class Labels
- **File:** `classes.npy`
- **Size:** ~1 KB
- **Contains:** Array of 54 sign IDs

---

## Model Specifications

### Medical Model
```
Input Shape: (30, 1662)
Output Shape: (54,)
Parameters: ~270K
Framework: TensorFlow 2.15 / Keras 3.0
```

### General Model
```
Input Shape: (126,)
Output Shape: (25,)
Parameters: ~50K
Framework: TensorFlow 2.15 / Keras 3.0
```

---

## Loading Models

```python
import tensorflow as tf

# Load medical model
med_model = tf.keras.models.load_model('medical_model.h5')

# Load general model
gen_model = tf.keras.models.load_model('general_model.h5')

# Load class labels
import numpy as np
classes = np.load('classes.npy', allow_pickle=True)
```

---

## Reproducibility

Models were trained with:
- Random seed: 42
- TensorFlow 2.15
- Python 3.10
- CUDA 12.0 (if GPU used)
