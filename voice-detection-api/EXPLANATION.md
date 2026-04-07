# Voice Detection Components Explanation

## 1. Explainable AI (XAI) Component

The Explainable AI (XAI) feature bridges the gap between raw probability scores and human understanding by generating plain-text explanations. It extracts explicit acoustic features from the audio using the `librosa` library and maps these metrics to human-readable reasons in the `extract_features_and_explain` function.

### How it works:
1. **Pitch Stability:** It tracks the pitch over time using `librosa.piptrack`. Human speech naturally fluctuates in pitch due to emotion and prosody. If the pitch standard deviation is unnaturally low (< 20Hz variation), it signifies robotic or monotone synthesis, yielding the explanation: *"Voice pitch is extremely stable"*.
2. **Spectral Variance:** It measures spectral flatness. Synthetic voices often have repetitive or abnormally flat spectral signatures. High flatness triggers the explanation: *"Spectral variation typical of synthetic voices"*.
3. **Pauses & Cadence:** Using `librosa.effects.split`, it calculates the ratio of silence vs active speech. Deepfake models often struggle to generate natural breathing pauses, resulting in unbroken speech blocks. A silence ratio of < 10% generates: *"Low natural pauses detected"*.
4. **Energy Variation:** It measures the Root Mean Square (RMS) energy. Human speech has dynamic ranges of loud and soft syllables. Lack of variance signals synthetic origin: *"Unnaturally low energy variation"*.

These reasons are appended to an array and sent directly to the frontend, allowing users to understand *why* the AI generated its score, instead of just receiving a number.

---

## 2. Language Detection Component

The Language Detection component utilizes a local `faster-whisper` Model, which provides highly accurate speech-to-text and language identification.

### How it works:
1. **Audio Pre-processing:** Whisper models are strictly trained on 16kHz audio. To ensure accuracy, any incoming audio (wav, mp3, flac) is forcefully resampled in `utils.py` to 16kHz (`sr=16000`).
2. **Volume Normalization:** The audio vector is normalized to ensure soft speech doesn't trigger false negatives.
3. **Model Inference:** A compact `"base"` Whisper model runs inference locally (ensuring fast performance). A targeted prompt is provided (`initial_prompt="English, Hindi, Spanish, French, German, Chinese, Japanese, Arabic, Russian, Portuguese"`) to bias the model's language space toward the 10 requested languages.
4. **Clean Mapping:** The raw short-code returned by Whisper (e.g., `"es"`, `"ja"`) is cleanly mapped to our backend schema (`SupportedLanguage.SPANISH`, `SupportedLanguage.JAPANESE`), ensuring the final JSON sent to the frontend is perfectly structured and safe for your UI.
