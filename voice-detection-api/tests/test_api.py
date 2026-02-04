import requests
import base64
import json
import numpy as np
import soundfile as sf
import io

def generate_sine_wave(duration=2.0, sample_rate=22050, frequency=440.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate a pure sine wave (should trigger "Artificial/Monotone" or just look clean)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    return audio, sample_rate

def create_base64_audio():
    audio, sr = generate_sine_wave()
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def test_api():
    url = "http://127.0.0.1:8000/predict"
    b64_audio = create_base64_audio()
    
    payload = {
        "audio_data": b64_audio
    }
    
    headers = {
        "x-api-key": "test-key-123", # Matching the key used in server start command
        "Content-Type": "application/json"
    }
    
    print("Sending request to API...")
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("Error:", response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()
