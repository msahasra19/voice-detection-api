import os
import requests
import numpy as np
import scipy.io.wavfile as wav

def download_file(url, folder, prefix=""):
    os.makedirs(folder, exist_ok=True)
    local_filename = os.path.join(folder, f"{prefix}{url.split('/')[-1]}")
    if not local_filename.endswith(".wav"):
        local_filename += ".wav"
        
    print(f"Downloading {url} to {local_filename}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, stream=True, timeout=30, headers=headers)
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1000:
            print(f"Successfully saved {local_filename} ({os.path.getsize(local_filename)} bytes)")
            return local_filename
        else:
            print(f"File {local_filename} is too small or missing.")
            return None
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def setup_dataset():
    base_dir = "dataset"
    
    # Language Samples (trying to find stable ones)
    lang_urls = [
        ("en", "https://www.voiptroubleshooter.com/open_speech/american/OSR_us_000_0010_8k.wav"),
        # Using some alternative links for Hindi/Telugu from public sources
        ("hi", "https://raw.githubusercontent.com/talrejaaditya/Hindi-Speech-Recognition/master/Checkpoints/1.wav"),
        ("te", "https://raw.githubusercontent.com/Sreeram02/Telugu-Speech-Dataset/main/Telugu_Audio_Samples/sample1.wav")
    ]

    print("\n--- Downloading Language Samples (EN, HI, TE) ---")
    for code, url in lang_urls:
        download_file(url, os.path.join(base_dir, "languages"), prefix=f"{code}_")

    # AI Samples
    print("\n--- Generating Synthetic AI Samples ---")
    os.makedirs(os.path.join(base_dir, "ai"), exist_ok=True)
    # (Simplified generation)
    sr = 16000
    t = np.linspace(0, 3, sr*3)
    for i in range(3):
        sig = np.sin(2*np.pi*440*t) + np.random.normal(0, 0.1, len(t))
        wav.write(os.path.join(base_dir, "ai", f"synthetic_{i}.wav"), sr, (sig*32767).astype(np.int16))

if __name__ == "__main__":
    setup_dataset()
