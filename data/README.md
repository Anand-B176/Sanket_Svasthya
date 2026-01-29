# Dataset Information

## 📊 Overview

This project uses the **Medical Sign Language Dataset** for healthcare communication.

---

## 📁 Dataset Sources

### Primary Dataset
- **Name:** Indian Medical Sign Language Dataset
- **Source:** Provided by competition organizers
- **Signs:** 55 medical signs (54 available, Sign_30 missing)
- **Performers:** 16 performers × 2 takes each
- **Format:** Video frames (JPG) + Kinect skeleton (CSV)

### Secondary Dataset (ISL Alphabets)
- **Name:** Indian Sign Language Alphabet Dataset
- **Signs:** 25 letters (A-Y, excluding J)
- **Format:** Pre-extracted NumPy arrays

---

## 📈 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Signs | 54 (medical) + 25 (alphabets) |
| Total Samples | 1,728 (medical) + 625 (alphabets) |
| Performers | 16 |
| Takes per Performer | 2 |
| Frames per Sample | 50-150 (variable) |
| Feature Dimension | 1,662 (MediaPipe Holistic) |

---

## 📂 Data Organization

```
Original Dataset Structure:
Signs(1-5)/
├── Sign_01_Performer_01_1/
│   ├── 01 Times/
│   ├── 02 Color Frames/        # ← Used for feature extraction
│   ├── 03 Infrared Frames/
│   ├── 04 Depth Frames/
│   ├── 05 BodyIndex Frames/
│   ├── 06 Body Skels data/     # Kinect skeleton (not used)
│   └── 07 Color Body Frames/
...
```

---

## 🔄 Preprocessing Pipeline

1. **Extract Color Frames** from dataset folders
2. **Process with MediaPipe Holistic** → 1,662 features/frame
3. **Apply Nose-Centered Normalization** → Translation invariance
4. **Pad/Truncate to 30 frames** → Fixed sequence length
5. **Save as NumPy arrays** → `.npy` files

---

## ⚠️ Data Notes

- **Sign_30 is missing** from the original dataset
- Model trained on 54 classes instead of 55
- Augmentation applied (3× dataset size)
- 80/20 train/test split with stratification

---

## 📥 Download Instructions

The processed data is too large for GitHub. Download from:

**Google Drive:** [Link to be added after upload]

After downloading, place files in:
```
data/processed/
├── X_train.npy
├── y_train.npy
├── X_test.npy
├── y_test.npy
└── classes.npy
```
