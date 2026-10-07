import requests, base64, os, glob

def test_file(path, label):
    if not os.path.exists(path):
        print(f'{label}: File not found {path}')
        return
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = requests.post('http://127.0.0.1:8000/predict', json={'audio_base64': b64})
    if resp.status_code == 200:
        data = resp.json()
        print(f'[{label}] => Verdict: {data.get("classification")} | Deepfake Risk: {data.get("deepfake_risk_score", 0):.2f} | Language: {data.get("detected_language")} ({data.get("language_confidence", 0)*100:.1f}%) | Latency: {data.get("processing_time_ms")}ms')
        print(f'  Reasons: {data.get("explainability")[:2]}')
        print(f'  Segments count: {len(data.get("segments", []))}')
    else:
        print(f'Error {resp.status_code}: {resp.text}')

human_files = glob.glob('dataset/human/*.wav')
ai_files = glob.glob('dataset/ai/*.wav')

if human_files:
    test_file(human_files[0], 'HUMAN SAMPLE')
if ai_files:
    test_file(ai_files[0], 'AI SYNTHETIC SAMPLE')
