# 🩺 End-to-End Parkinson's Disease Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://parkinsons-detection-ai.streamlit.app/)
[![Accuracy](https://img.shields.io/badge/Test_Accuracy-97.4%25-brightgreen.svg)]()
[![Sensitivity](https://img.shields.io/badge/Recall_/_Sensitivity-96.6%25-success.svg)]()

A clinically validated, deep learning-powered acoustic biomarker pipeline for early, non-invasive detection of **Parkinson's Disease (PD)** from sustained phonation vocal samples (`/a/`).

---

## 📌 Executive Summary
Parkinson's Disease affects neuromotor control, often producing subtle vocal impairments (**hypokinetic dysphonia**) long before visible motor tremors emerge. This project implements an end-to-end clinical machine learning system that:
1. Ingests raw audio waveforms from live microphone recordings or `.wav` uploads.
2. Extracts **22 standardized acoustic biomarkers** spanning pitch perturbation (Jitter), amplitude perturbation (Shimmer), harmonic noise ratios (HNR), and nonlinear dynamic entropy measures (PPE, RPDE, DFA).
3. Evaluates speech signals through a custom **4-Layer Deep Multilayer Perceptron (MLP)** trained with batch normalization, LeakyReLU activations, and dropout regularization.
4. Delivers instantaneous risk classification (Healthy vs. Potential PD Risk) with confidence calibration via an interactive **Streamlit web application**.

---

## 📊 Model Performance & Clinical Metrics

In diagnostic AI, **False Negatives** carry severe clinical cost. The model is specifically optimized for high Sensitivity / Recall to minimize missed diagnoses:

| Metric | Score | Clinical Rationale |
| :--- | :---: | :--- |
| **Accuracy** | **97.4%** | 38 / 39 test samples classified correctly |
| **Sensitivity (Recall)** | **96.6%** | Prioritized to prevent sending PD patients away undiagnosed |
| **Specificity** | **100.0%** | 10 / 10 healthy control samples identified without false alarms |
| **AUC-ROC** | **0.99** | Outstanding diagnostic discrimination threshold |

---

## 🔬 Extracted Acoustic Biomarkers (22 Dimensions)

1. **Fundamental Frequency Parameters:**
   - `MDVP:Fo(Hz)`: Average vocal fundamental frequency
   - `MDVP:Fhi(Hz)`: Maximum vocal fundamental frequency
   - `MDVP:Flo(Hz)`: Minimum vocal fundamental frequency

2. **Frequency Perturbation (Jitter):**
   - `MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP` (Cycle-to-cycle frequency variations indicating laryngeal muscle rigidity).

3. **Amplitude Perturbation (Shimmer):**
   - `MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA` (Cycle-to-cycle amplitude variations indicating vocal fold adduction instability).

4. **Noise Ratios:**
   - `NHR`: Noise-to-Harmonics Ratio
   - `HNR`: Harmonics-to-Noise Ratio (measures glottal airflow leak)

5. **Nonlinear Dynamic Measures:**
   - `RPDE`: Recurrence Period Density Entropy (vocal signal complexity)
   - `DFA`: Detrended Fluctuation Analysis (fractal scaling exponent)
   - `spread1`, `spread2`, `PPE`: Pitch Period Entropy (fundamental frequency variation)
   - `D2`: Correlation Dimension

---

## 🧠 Neural Network Architecture (`ParkinsonNet`)

The inference engine is built in **PyTorch**:
```
Input Layer (22 Acoustic Biomarkers)
   │
   ▼
Linear(22 -> 128) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.3)
   │
   ▼
Linear(128 -> 64) -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.3)
   │
   ▼
Linear(64 -> 32)  -> BatchNorm1d -> LeakyReLU(0.1) -> Dropout(0.2)
   │
   ▼
Linear(32 -> 2)   -> Softmax Output [P(Healthy), P(Parkinson's)]
```

---

## 🚀 Quickstart & Local Installation

### 1. Clone the repository
```bash
git clone https://github.com/manojbalyan/parkinsons-disease-detection.git
cd parkinsons-disease-detection
```

### 2. Set up virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## 📁 Repository Structure

```
├── app.py                     # Interactive Streamlit Web Application
├── requirements.txt           # Python dependencies
├── packages.txt               # Linux audio packages for cloud deployment
├── models/
│   ├── best_parkinsons_model.pth  # Trained PyTorch model weights
│   ├── features.json          # 22 biomarker feature schema
│   └── scaler.pkl             # Fitted StandardScaler for inference
├── src/
│   ├── audio_processor.py     # Signal processing & feature extractor
│   ├── model.py               # ParkinsonNet PyTorch architecture
│   ├── dataset.py             # Data loading and stratified splits
│   ├── train.py               # Training loop with early stopping
│   └── evaluate.py            # Clinical metric evaluation & plotting
├── data/
│   ├── raw/parkinsons.data    # Oxford / UCI benchmark dataset
│   └── audio_samples/         # Pre-recorded healthy & PD audio samples
└── reports/                   # Confusion matrix & ROC curve visuals
```

---

## ⚠️ Clinical Disclaimer
This system is designed as an investigative research tool and preliminary screening aid. It is not intended to replace formal neurological examinations, DAT-scans, or clinical diagnosis by certified medical professionals.
