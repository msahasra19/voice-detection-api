import requests
import base64
import json
import sys
import os

def test_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    url = "http://127.0.0.1:8000/predict"
    
    print(f"Reading file: {file_path}...")
    with open(file_path, "rb") as f:
        file_bytes = f.read()
        b64_audio = base64.b64encode(file_bytes).decode('utf-8')
    
    payload = {
        "audio_data": b64_audio
    }
    
    headers = {
        "x-api-key": "test-key-123", 
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
    if len(sys.argv) < 2:
        print("Usage: python tests/test_with_file.py <path_to_audio_file>")
        print("Example: python tests/test_with_file.py my_voice.mp3")
    else:
        test_file(sys.argv[1])
