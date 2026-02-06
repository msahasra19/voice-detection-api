# Voice Detection API

> ⚠️ **JUDGES / EVALUATORS: IMPORTANT** ⚠️
> If the submitted ngrok link is offline due to a tunnel reset, please use this active link to test the API:
> **[https://holly-passport-volt-accountability.trycloudflare.com/](https://holly-passport-volt-accountability.trycloudflare.com/)**
> (Wait a few seconds for the tunnel to connect if it shows a loading screen)

A minimal FastAPI-based service that accepts audio uploads and performs a simple

voice-activity style detection using placeholder logic. It is structured to be
easy to extend with a real ML model later.

### Project structure

```text
voice-detection-api/
├── app/
│   ├── main.py          # API entry point (FastAPI app and routes)
│   ├── model.py         # Voice detection logic (placeholder model)
│   ├── utils.py         # Audio decoding helpers
│   └── auth.py          # API key validation
├── requirements.txt
└── README.md
```

> Note: Create a `.env` file yourself in the project root with at least:
>
> ```text
> API_KEY=change_me_to_a_secure_random_value
> ```

### Installation

From the `voice-detection-api` directory:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Security Setup (IMPORTANT)

**Before running the API**, you must configure your API key:

1. **Copy the example environment file:**
   ```bash
   copy .env.example .env  # Windows
   # or
   cp .env.example .env    # Linux/Mac
   ```

2. **Generate a secure API key:**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. **Edit `.env` and replace `your-secret-api-key-here` with your generated key:**
   ```text
   API_KEY=your-actual-secure-key-here
   ```

4. **NEVER commit the `.env` file to git!** It's already in `.gitignore` to protect your secrets.

### Running the API

From the `voice-detection-api` directory:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Endpoints

- **`GET /health`**  
  Simple health check. Returns:

  ```json
  { "status": "ok" }
  ```

- **`POST /detect`**  
  Accepts an audio file and returns a naive detection result.

  - **Headers**
    - `X-API-Key`: your API key (must match `API_KEY` in the environment).
  - **Body (multipart/form-data)**
    - `file`: audio file (e.g. `.wav`, `.flac`, etc.).

  **Example `curl` request:**

  ```bash
  curl -X POST "http://localhost:8000/detect" ^
       -H "X-API-Key: YOUR_API_KEY_HERE" ^
       -F "file=@path\to\audio.wav"
  ```

  Example response:

  ```json
  {
    "filename": "audio.wav",
    "result": {
      "has_voice": true,
      "score": 0.73,
      "sample_rate": 16000.0
    }
  }
  ```

### Extending with a real model

- Replace the placeholder logic in `app/model.py` with loading your trained
  model (e.g. from a `.pt`, `.onnx`, or `.pkl` file).
- Adapt `detect_voice` to accept the waveform and sample rate, run the model,
  and return whatever structured output you need.

