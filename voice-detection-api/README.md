# Voice Detection API (ML-Based)

This project has been updated to use **Local Machine Learning Models** and requires **No External API Keys**.

### Key Features
- **Local Language Detection**: Uses `faster-whisper` (tiny model) to detect languages (English, Hindi, Tamil, Telugu, Malayalam) locally without calling Google APIs.
- **Local Heuristic Analysis**: Detects AI vs Human voices using advanced signal processing (Spectral Flatness, Pitch Stability, Zero Crossing Rate).
- **Public Access**: Authentication key requirements have been removed for easier local testing and evaluation.

### Project structure

```text
voice-detection-api/
├── app/
│   ├── main.py          # API entry point (Public /predict endpoint)
│   ├── analysis.py      # Local ML logic (Whisper + Signal Processing)
│   ├── model.py         # Orchestration of detection
│   ├── utils.py         # Audio decoding helpers
│   └── schemas.py       # Pydantic models for request/response
├── requirements.txt
└── README.md
```

### Installation

From the `voice-detection-api` directory:

```bash
# It is recommended to use a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
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
  - **Headers**: No API Key required.
  - **Body**: JSON with `audio_url` or `audio_base64`.

### AI vs Human Detection
The detection uses local heuristics including:
1. **Spectral Flatness**: Detects synthetic "robotic" buzz.
2. **Pitch Stability**: Identifies monotone synthesis.
3. **SNR Analysis**: Identifies studio-quality audio typical of high-end AI (like ElevenLabs).

