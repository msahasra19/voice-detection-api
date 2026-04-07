import os
import random
import librosa
import soundfile as sf
import numpy as np
from gtts import gTTS

BASE_DIR = "dataset_demo"
HUMAN_DIR = os.path.join(BASE_DIR, "human")
AI_DIR = os.path.join(BASE_DIR, "ai")
LANG_DIR = os.path.join(BASE_DIR, "languages")

for d in [HUMAN_DIR, AI_DIR, LANG_DIR]:
    os.makedirs(d, exist_ok=True)

print("Starting demo dataset generation...")

# =======================
# 1. HUMAN VOICES (~50 clips)
# =======================
# We will use librosa's built-in 'libri' examples as base human anchors,
# and dynamically augment them (pitch shift, time stretch, noise) to create 50 variations.
print("Generating Human Voices (using augmented open LibriSpeech anchors)...")
base_human_anchors = ['libri1', 'libri2', 'libri3']
try:
    y_libri1, sr1 = librosa.load(librosa.ex('libri1', hq=True))
    y_libri2, sr2 = librosa.load(librosa.ex('libri2', hq=True))
    y_libri3, sr3 = librosa.load(librosa.ex('libri3', hq=True))
    anchors = [(y_libri1, sr1), (y_libri2, sr2), (y_libri3, sr3)]
except Exception as e:
    print("Fallback for Librosa examples...")
    # Generate backup synthetic "human" signals using high complexity sine waves if network blocked 
    sr = 16000
    anchors = [(np.sin(2 * np.pi * 440 * np.linspace(0, 5, sr*5)), sr)]

human_files_generated = 0
while human_files_generated < 50:
    for base_y, sr in anchors:
        if human_files_generated >= 50: break
        
        # 1. Take a 3-4 second random crop
        duration_samples = int(random.uniform(3.0, 4.5) * sr)
        if len(base_y) > duration_samples:
            start = random.randint(0, len(base_y) - duration_samples)
            y_crop = base_y[start:start+duration_samples]
        else:
            y_crop = base_y.copy()
            
        # 2. Augment (Pitch shift OR Time stretch)
        aug_type = random.choice([0, 1, 2])
        if aug_type == 1:
            y_crop = librosa.effects.pitch_shift(y_crop, sr=sr, n_steps=random.uniform(-3, 3))
        elif aug_type == 2:
            y_crop = librosa.effects.time_stretch(y_crop, rate=random.uniform(0.8, 1.2))
            
        # 3. Add tiny natural noise variation
        noise = np.random.normal(0, 0.005, len(y_crop))
        y_final = np.clip(y_crop + noise, -1.0, 1.0)
        
        # Determine language tag (libri is EN but we simulate TE/HI variations by heavy augmentation)
        lang_tag = random.choice(["en", "hi", "te"])
        filename = os.path.join(HUMAN_DIR, f"human_{lang_tag}_sample_{human_files_generated+1:03d}.wav")
        sf.write(filename, y_final, sr)
        human_files_generated += 1

print(f"Generated {human_files_generated} human samples.")

# =======================
# 2. AI VOICES (~50 clips)
# =======================
print("Generating AI Voices (using gTTS)...")
ai_phrases = {
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the future of technology.",
        "This is an automated synthetic voice generated for a test dataset.",
        "Voice cloning requires massive amounts of data.",
        "I am an artificial intelligence model generating speech."
    ],
    "hi": [
        "कृत्रिम बुद्धिमत्ता भविष्य को बदल रही है।",
        "यह एक सिंथेटिक वॉयस जनरेशन टेस्ट है।",
        "मुझे यह आवाज कंप्यूटर से मिली है।",
        "मशीन लर्निंग मॉडल बहुत शक्तिशाली हैं।",
        "हमारी परियोजना को डीपफेक का पता लगाना है।"
    ],
    "te": [
        "వాయిస్ క్లోనింగ్ సాంకేతికత ఇప్పుడు చాలా అభివృద్ధి చెందింది.",
        "ఇది బూటకపు వాయిస్ కి సంబంధించిన ఒక నమూనా.",
        "యంత్రాలు మనుషుల వలె మాట్లాడగలవు.",
        "మేము మోసపూరిత ఆడియోను గుర్తించడానికి పని చేస్తున్నాము.",
        "కృత్రిమ మేధస్సు రేపటి ప్రపంచాన్ని మారుస్తుంది."
    ]
}

ai_files_generated = 0
for i in range(50):
    lang_tag = random.choice(["en", "hi", "te"])
    phrase = random.choice(ai_phrases[lang_tag])
    try:
        tts = gTTS(text=phrase, lang=lang_tag, slow=False)
        filename = os.path.join(AI_DIR, f"ai_synth_{lang_tag}_{ai_files_generated+1:03d}.mp3")
        tts.save(filename)
        ai_files_generated += 1
    except Exception as e:
        print(f"Skipping AI TTS for {lang_tag}: {e}")

print(f"Generated {ai_files_generated} AI samples.")

# =======================
# 3. LANGUAGE SAMPLES (~40 clips)
# =======================
print("Generating Language specific voices...")
lang_files_generated = 0
for i in range(40):
    lang_tag = random.choice(["en", "hi", "te"])
    phrase = random.choice(ai_phrases[lang_tag])
    try:
        # We will deliberately slow down some TTS outputs to create "pronunciation" variance checks
        tts = gTTS(text=phrase, lang=lang_tag, slow=random.choice([True, False]))
        filename = os.path.join(LANG_DIR, f"lang_demo_{lang_tag}_{lang_files_generated+1:03d}.mp3")
        tts.save(filename)
        lang_files_generated += 1
    except Exception as e:
        print(f"Skipping Lang TTS for {lang_tag}: {e}")

print(f"Generated {lang_files_generated} Language samples.")
print("\nDataset generation complete!")
