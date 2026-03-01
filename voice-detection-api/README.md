# Voice Detection ML (Local Processing)

This repository contains a local, machine learning-based API for detecting AI-generated vs. Human voices and identifying languages. It uses the `faster-whisper` model and custom signal processing heuristics, requiring **No External API Keys**.

### Key Features
- **Local Language Detection**: Uses `faster-whisper` (tiny model) to detect languages (English, Hindi, Tamil, Telugu, Malayalam) locally.
- **AI vs Human Detection**: 
    - **Heuristic Analysis**: Spectral Flatness, Pitch Stability, and SNR analysis.
    - **Trained Model**: Extensible Random Forest classifier for audio feature extraction (MFCCs, Spectral Contrast).
- **Public Access**: Authentication key requirements removed for streamlined development and testing.

### Project Structure

```text
voice-detection-api/
├── app/
│   ├── main.py          # API entry point (Public /predict endpoint)
│   ├── analysis.py      # Core ML logic (Whisper + Signal Processing)
│   ├── model.py         # Orchestration of detection
│   ├── utils.py         # Audio decoding helpers
│   └── voice_model.joblib # Trained Random Forest model (if applicable)
├── training/
│   ├── train_model.py   # Training script for AI vs Human detection
│   └── download_dataset.py # Helper to prepare training data
├── requirements.txt
└── README.md
```

### Installation

From the project root:

```bash
# Recommended: Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate # Linux/Mac

pip install -r requirements.txt
```

### Running the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
*Note: On the first run, the system will automatically download the Whisper 'tiny' model (~75MB).*

### Endpoints

- **`GET /health`**: Status check.
- **`POST /predict`**: Detect AI vs Human voice + Language.
  - **Body**: JSON with `audio_url` or `audio_base64`.

### Training the Model

To improve AI vs Human detection accuracy, you can train the local classifier:
1. Place your audio samples in `dataset/human/` and `dataset/ai/`.
2. Run the training script:
   ```bash
   python training/train_model.py
   ```
3. The new model will be saved to `app/voice_model.joblib`.

### AI vs Human Detection Heuristics
1. **Spectral Flatness**: Identifies the "noisy" nature of synthesis.
2. **Pitch Stability**: Detects the unnatural consistency of AI voices.
3. **MFCC Analysis**: Captures timbre differences between human vocal tracts and synthetic vocoders.


