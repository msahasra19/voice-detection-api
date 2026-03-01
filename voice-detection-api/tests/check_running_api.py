import requests
import base64
import os
import json

def test_samples():
    url = "http://127.0.0.1:8000/predict"
    
    # 1. Human Sample
    human_path = os.path.join("dataset", "human", "test.wav")
    # 2. AI Sample
    ai_path = os.path.join("dataset", "ai", "synthetic_ai_0.wav")

    test_cases = [
        ("HUMAN (Real)", human_path),
        ("AI (Synthetic)", ai_path)
    ]

    for label, path in test_cases:
        print(f"\n--- Testing {label} ---")
        if not os.path.exists(path):
            print(f"Skipping: {path} not found.")
            continue
            
        with open(path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
            
        payload = {"audio_data": b64_data}
        try:
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(f"Classification: {data['classification']}")
                print(f"Risk Score: {data['deepfake_risk_score']}")
                print(f"Language: {data['detected_language']}")
                print(f"Reasons: {data['explainability']}")
            else:
                print(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    test_samples()
