# Speech Emotion Recognition using MLP & Acoustic Features

A machine learning pipeline for classifying human emotional states from speech audio using acoustic feature extraction (MFCC, Pitch, RMS Energy) and a Multi-Layer Perceptron classifier — achieving **79.49% accuracy** across 8 emotion categories on the RAVDESS dataset.

---

## Overview

This project explores Speech Emotion Recognition (SER) by combining signal processing techniques with supervised machine learning. For each audio sample, the pipeline:

1. Loads and normalizes the raw waveform
2. Visualizes the signal in time domain, frequency domain (FFT), and spectrogram
3. Extracts acoustic features: 40 MFCCs, average pitch (F0), and RMS energy
4. Trains an MLPClassifier (128 → 64 hidden units) with StandardScaler normalization
5. Evaluates using accuracy, classification report, and confusion matrix

**Dataset:** [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://zenodo.org/record/1188976) a validated, actor-recorded dataset with 8 emotion labels.

---

## Emotion Classes

| Code | Label    |
|------|----------|
| 01   | Neutral  |
| 02   | Calm     |
| 03   | Happy    |
| 04   | Sad      |
| 05   | Angry    |
| 06   | Fearful  |
| 07   | Disgust  |
| 08   | Surprised |

---

## Results

| Metric        | Score  |
|---------------|--------|
| Accuracy      | 79.49% |
| Macro F1      | 0.79   |
| Weighted F1   | 0.79   |

**Per-class F1 scores:**

| Emotion   | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Disgust   | 0.67      | 0.80   | 0.73     |
| Surprised | 1.00      | 0.50   | 0.67     |
| Angry     | 0.83      | 1.00   | 0.91     |
| Neutral   | 0.83      | 1.00   | 0.91     |
| Calm      | 0.60      | 0.60   | 0.60     |
| Sad       | 0.80      | 0.80   | 0.80     |
| Happy     | 1.00      | 0.80   | 0.89     |
| Fearful   | 0.80      | 0.80   | 0.80     |

---

## Project Structure

```
├── Main.ipynb          # Full pipeline: EDA → Feature Extraction → Modeling
├── Audio/              # RAVDESS audio samples (see Dataset Setup below)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Dataset Setup

The audio files used in this project are sourced from the **RAVDESS dataset** and are **not included** in this repository due to file size constraints.

**Download the dataset:**

```bash
# Option 1: Download directly from Zenodo
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip

# Option 2: Via Kaggle (requires Kaggle CLI)
kaggle datasets download -d uwrfkaggler/ravdess-emotional-speech-audio
```

After downloading, place all `.wav` files into an `Audio/` folder at the root of this repository.

The RAVDESS filename convention encodes metadata as:
```
03-01-[emotion]-[intensity]-[statement]-[repetition]-[actor].wav
```
The third segment (position index 2) maps to the emotion label.

---

## Installation

```bash
git clone https://github.com/Bimzt/speech-emotion-recognition-mlp.git
cd speech-emotion-recognition-mlp
pip install -r requirements.txt
```

Then open and run the notebook:

```bash
jupyter notebook Main.ipynb
```

---

## Requirements

See `requirements.txt`. Core dependencies:

- `librosa` — audio loading, MFCC, pitch (pyin), STFT
- `scikit-learn` — MLPClassifier, StandardScaler, evaluation metrics
- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — visualization
- `scipy` — FFT computation

Python version: **3.12**

---

## Methodology

### Feature Extraction

For each audio file, the following features are extracted and concatenated into a 42-dimensional feature vector:

| Feature         | Dimension | Description                              |
|-----------------|-----------|------------------------------------------|
| MFCC (mean)     | 40        | Mel-frequency cepstral coefficients, averaged over time |
| RMS Energy      | 1         | Mean root-mean-square energy (loudness proxy) |
| Avg Pitch (F0)  | 1         | Mean fundamental frequency via pYIN algorithm |

### Model

- **Architecture:** MLPClassifier with hidden layers `(128, 64)`, ReLU activation, Adam optimizer
- **Preprocessing:** StandardScaler on training split, applied to test split
- **Split:** 80% train / 20% test, stratified by emotion class, `random_state=42`

---

## Limitations & Future Work

- Dataset is English-language only (RAVDESS); cross-lingual generalization is untested
- Small test set (39 samples) — results may vary with different random seeds
- No cross-validation; a single train/test split limits statistical reliability
- Future improvements: add ZCR, Spectral Centroid, and Chroma features; experiment with CNN/LSTM on raw spectrograms; implement k-fold cross-validation

---

## Acknowledgements

- Livingstone SR, Russo FA (2018) *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*. PLOS ONE.
- [librosa](https://librosa.org/) — audio and music signal analysis library
