# Lightweight Privacy-Preserving Voice Anti-Spoofing & Multilingual Speech Analysis System

An end-to-end, CPU-friendly audio forensics platform that performs two core tasks:
1. **Deepfake Speech Detection**: Distinguishing bona fide human recordings from AI-generated, neural TTS, voice cloned, and vocoder-synthesized speech using supervised machine learning combined with interpretable acoustic indicators and calibrated decision fusion.
2. **Multilingual Language Identification**: Zero-shot local language identification and transcript preview using a local, quantized (`int8`) **Faster Whisper** model (no cloud APIs required).
3. **Suspicious Segment Localization**: Second-by-second temporal window analysis generating an interactive risk timeline capable of localizing partially manipulated and spliced deepfake audio.

---

## 🏛️ System Architecture

```
                          AUDIO INPUT (MP3 / WAV / FLAC / M4A)
                                           │
                                           ▼
                            PREPROCESSING & NORMALIZATION
                          Mono • 16 kHz Resample • Peak Normalize
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    ▼                      ▼                      ▼
           DEEPFAKE ML BRANCH       ACOUSTIC BRANCH        LANGUAGE BRANCH
                    │                      │                      │
            97-dim Acoustic         SNR (dB), F0 Pitch     Quantized Faster
            Vector: MFCCs (μ, σ,    Stability, Silence     Whisper (int8 on CPU)
            Δ, Δ²), Flatness,       Dynamics, Quality             │
            Spectral Contrast              │                      │
                    │                      │                      ▼
                    ▼                      ▼               SPOKEN LANGUAGE
              RANDOM FOREST         ACOUSTIC RISK           IDENTIFICATION
               CLASSIFIER             HEURISTIC           & TRANSCRIPT SAMPLE
                    │                      │
                    └──────────┬───────────┘
                               ▼
                      LEARNED FUSION MODEL
                (Calibrated Logistic Regression)
                               │
                               ▼
               1-SECOND WINDOW TIMELINE ANALYSIS
                               │
                               ▼
              SUSPICIOUS SEGMENT LOCALIZATION & UI
```

---

## 🚀 Key Modules & Research Methodology

### 1. Acoustic Machine Learning Branch
- **Feature Extraction**: 97-dimensional acoustic feature vector extracting:
  - **13 MFCCs**: Static, Velocity ($\Delta$), and Acceleration ($\Delta^2$) means and standard deviations.
  - **Spectral Flatness**: Measures tonal vs noise energy distribution.
  - **Spectral Contrast across 5 Sub-bands**: Captures harmonic peak-to-valley energy ratios.
  - **Spectral Centroid & Rolloff**: Detects unnatural spectral smoothing.
  - **Zero Crossing Rate (ZCR) & Shimmer Proxies**: Measures micro-glottal irregularity.
- **Classifier**: Supervised Random Forest trained on bona fide and spoofed speech benchmarks.

### 2. Interpretable Physical Acoustic Signal Branch
- **Signal-to-Noise Ratio (SNR)**: Short-time Fourier transform (STFT) RMS energy estimation against the 10th-percentile noise floor (in dB).
- **Pitch & Fundamental Frequency (F0) Dynamics**: Fast YIN-based F0 tracking computing median pitch, standard deviation ($\sigma_{F0}$), and normalized pitch stability to flag robotic flat-pitch synthesis or erratic vocoder jumps.
- **Silence Dynamics**: Measures pause ratios and speech flow continuity.

### 3. Learned Decision Fusion
- Rather than relying on arbitrary fixed weights, a **Logistic Regression Fusion Model** dynamically combines the Random Forest probability with physical acoustic indicators to ensure robustness across both clean laboratory benchmarks and real-world noisy audio.

### 4. Second-by-Second Suspicious Segment Timeline
- Slices audio into 1-second temporal windows and evaluates each segment independently.
- Localizes partially manipulated or spliced audio where AI synthetic speech is mixed with authentic human recordings.

### 5. Multilingual Language Identification
- Utilizes local, quantized (`int8`) **Faster Whisper** for zero-latency language classification across English and Indian languages (Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Bengali, Gujarati, Punjabi).

---

## 📊 Experimental Evaluation & Ablation Study

Evaluated on controlled benchmarks (**ASVspoof 2019 LA** and **In the Wild** dataset protocols) across 4 experimental configurations:

| Config | System Architecture | Features | Classifier | ROC-AUC | Equal Error Rate (EER) |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **A** | MFCC + Random Forest | 26 MFCC features ($\mu, \sigma$) | Random Forest | 0.974 | 6.8% |
| **B** | MFCC + Spectral Features + RF | 97 Acoustic dims (MFCC $\Delta$, Contrast, Flatness) | Random Forest | 0.988 | 4.2% |
| **C** | Random Forest + Heuristic Rule | 97 Acoustic dims + Fixed Rule Weights | Hybrid RF + Heuristics | 0.985 | 4.5% |
| **D** | **Learned Fusion System (Proposed)** | ML Probs + SNR + Pitch Stability + Silence + Flatness | RF + Logistic Regression Fusion | **0.994** | **2.4%** |

*EER (Equal Error Rate): The operating threshold where False Acceptance Rate equals False Rejection Rate. Lower EER indicates superior anti-spoofing discrimination.*

---

## 📁 Repository Structure

```text
voice-detection-api/
├── app/
│   ├── main.py                  # FastAPI server with /predict, /predict-file, /ablation-report
│   ├── analysis.py              # Dual-branch feature extraction, acoustics, & Whisper logic
│   ├── model.py                 # Orchestration, decision fusion, & spliced speech aggregation
│   ├── schemas.py               # Pydantic data validation schemas
│   ├── utils.py                 # Multi-format audio decoding (MP3, WAV, FLAC, M4A, OGG, WebM)
│   ├── auth.py                  # Optional API authentication helper
│   ├── voice_model.joblib       # Trained Random Forest classifier
│   └── fusion_model.joblib      # Trained Learned Logistic Regression fusion model
├── static/
│   ├── index.html               # Modern dark glassmorphic forensics dashboard
│   ├── styles.css               # Responsive design system & animations
│   └── script.js                # Frontend audio recorder, presets, timeline & ablation UI
├── training/
│   ├── train_model.py           # Supervised training script for RF and Fusion models
│   ├── dataset_manager.py       # ASVspoof 2019 loader & multi-generator benchmark synthesizer
│   ├── ablation_and_evaluation.py # 4-stage ablation experiments & anti-spoofing EER calculator
│   └── ablation_results.json    # JSON report containing benchmark metrics
├── test_inference.py            # Automated end-to-end HTTP test suite
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone & Setup Environment
```bash
git clone https://github.com/msahasra19/voice-detection-ml.git
cd voice-detection-ml/voice-detection-api

# Create & activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Forensics Server
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Forensics Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Ablation Benchmark API**: [http://127.0.0.1:8000/ablation-report](http://127.0.0.1:8000/ablation-report)

---

## 🔌 API Endpoints

### 1. `POST /predict`
Submit Base64 encoded audio or an audio URL.
```json
{
  "audio_base64": "<base64_encoded_audio>"
}
```

### 2. `POST /predict-file`
Direct multipart/form-data audio file upload (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.webm`).

---

## 🔬 Re-running Training & Ablation Experiments

```bash
# 1. Prepare multi-class benchmark samples
python training/dataset_manager.py

# 2. Train Random Forest & Learned Logistic Regression Fusion models
python training/train_model.py

# 3. Run full 4-stage ablation experiments & generate EER metrics
python training/ablation_and_evaluation.py
```
