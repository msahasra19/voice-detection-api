import os
import glob
import numpy as np
import scipy.io.wavfile as wav
import librosa
from typing import List, Tuple, Dict, Any

class DatasetManager:
    """
    Manages dataset loading, partition protocols (ASVspoof 2019 LA, In The Wild),
    and synthetic speech generator for controlled benchmarking.
    """

    @staticmethod
    def parse_asvspoof_protocol(protocol_file: str, audio_dir: str) -> List[Tuple[str, int]]:
        samples = []
        if not os.path.exists(protocol_file):
            return samples

        with open(protocol_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    file_id = parts[1]
                    key = parts[4].lower()
                    label = 0 if key == 'bonafide' else 1
                    
                    audio_path = os.path.join(audio_dir, f"{file_id}.flac")
                    if not os.path.exists(audio_path):
                        audio_path = os.path.join(audio_dir, f"{file_id}.wav")

                    if os.path.exists(audio_path):
                        samples.append((audio_path, label))
        return samples

    @staticmethod
    def load_directory_dataset(base_dir: str) -> Tuple[List[str], List[int]]:
        files = []
        labels = []

        human_dirs = [os.path.join(base_dir, "human"), os.path.join(base_dir, "bonafide")]
        ai_dirs = [os.path.join(base_dir, "ai"), os.path.join(base_dir, "spoof"), os.path.join(base_dir, "synthetic")]

        for h_dir in human_dirs:
            if os.path.exists(h_dir):
                for ext in ('*.wav', '*.mp3', '*.flac', '*.ogg'):
                    for fpath in glob.glob(os.path.join(h_dir, ext)):
                        files.append(fpath)
                        labels.append(0)

        for a_dir in ai_dirs:
            if os.path.exists(a_dir):
                for ext in ('*.wav', '*.mp3', '*.flac', '*.ogg'):
                    for fpath in glob.glob(os.path.join(a_dir, ext)):
                        files.append(fpath)
                        labels.append(1)

        return files, labels

    @staticmethod
    def generate_synthetic_benchmark_dataset(output_dir: str = "dataset", samples_per_class: int = 50):
        """
        Generates advanced multi-class speech waveforms:
        - Bona Fide Human: natural vocal tract glottal turbulence, dynamic formant transitions, micro-jitter, room acoustics.
        - AI Neural TTS / Voice Clones: HiFi-GAN phase artifacts, pitch flattening, pristine zero-noise gating, spectral smoothing.
        """
        sr = 16000
        os.makedirs(os.path.join(output_dir, "human"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "ai"), exist_ok=True)

        print(f"Generating {samples_per_class} realistic Human speech and {samples_per_class} Neural AI samples...")
        np.random.seed(42)

        # 1. BONA FIDE HUMAN SPEECH SAMPLES
        for i in range(samples_per_class):
            dur = np.random.uniform(3.0, 6.0)
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)

            # Human vocal tract fundamental frequency (F0) with natural intonation & micro-jitter
            f0_base = np.random.uniform(110, 240)
            # Intonation macro contour
            f0_macro = f0_base + 30 * np.sin(2 * np.pi * np.random.uniform(0.8, 2.2) * t) + 12 * np.cos(2 * np.pi * np.random.uniform(2.5, 4.5) * t)
            # Natural cycle-to-cycle micro-jitter (0.8% - 1.5%)
            jitter = np.random.normal(0, f0_base * 0.012, len(t))
            f0_total = np.clip(f0_macro + jitter, 65, 450)
            phase = 2 * np.pi * np.cumsum(f0_total) / sr

            # Multiple human vocal tract formants (F1, F2, F3, F4)
            sig = (
                0.55 * np.sin(phase) +
                0.28 * np.sin(2 * phase) +
                0.16 * np.sin(3 * phase) +
                0.10 * np.sin(4 * phase) +
                0.06 * np.sin(5 * phase)
            )

            # Syllabic speech cadence with natural breathing pauses
            cadence = np.sin(2 * np.pi * np.random.uniform(2.2, 4.0) * t) ** 2
            # Add sporadic natural pause intervals
            pause_mask = np.ones_like(t)
            num_pauses = np.random.randint(1, 4)
            for _ in range(num_pauses):
                p_start = np.random.uniform(0.5, dur - 1.0)
                p_len = np.random.uniform(0.2, 0.6)
                pause_mask[(t >= p_start) & (t <= p_start + p_len)] = 0.02

            sig = sig * cadence * pause_mask

            # Natural glottal breath turbulence and real-room ambient noise floor
            breath = np.random.normal(0, 0.025, len(t)) * cadence
            room_noise = np.random.normal(0, np.random.uniform(0.015, 0.035), len(t))
            sig = sig + breath + room_noise

            sig = sig / (np.max(np.abs(sig)) + 1e-7) * 0.88
            out_file = os.path.join(output_dir, "human", f"human_speech_{i+1:03d}.wav")
            wav.write(out_file, sr, (sig * 32767).astype(np.int16))

        # 2. AI SYNTHETIC / NEURAL TTS / VOICE CLONE SAMPLES
        for i in range(samples_per_class):
            dur = np.random.uniform(3.0, 6.0)
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)

            model_type = i % 4

            if model_type == 0:
                # Modern Neural TTS (ElevenLabs / OpenAI style): smooth intonation, pristine zero noise floor, harmonic purity
                f0_base = np.random.uniform(130, 220)
                # Overly smooth F0 curve (zero natural micro-jitter)
                f0_smooth = f0_base + 18 * np.sin(2 * np.pi * 1.2 * t)
                phase = 2 * np.pi * np.cumsum(f0_smooth) / sr
                sig = 0.65 * np.sin(phase) + 0.32 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
                cadence = np.sin(2 * np.pi * 3.2 * t) ** 2
                sig = sig * cadence
                # Pristine silence floor (0.0001 noise floor)
                sig = sig + np.random.normal(0, 0.0005, len(t))

            elif model_type == 1:
                # Neural Vocoder (HiFi-GAN / MelGAN artifacts): upper band phase buzz & high spectral flatness
                f0 = np.random.uniform(140, 200)
                phase = 2 * np.pi * f0 * t
                sig = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase)
                # Vocoder high-frequency carrier buzz (3kHz - 6kHz)
                vocoder_buzz = 0.12 * np.sin(2 * np.pi * np.random.uniform(3200, 5800) * t)
                cadence = np.sin(2 * np.pi * 3.5 * t) ** 2
                sig = (sig + vocoder_buzz) * cadence + np.random.normal(0, 0.001, len(t))

            elif model_type == 2:
                # Voice Conversion (VC): formant distortion, discontinuous pitch transitions
                f0 = 170 + 60 * np.sign(np.sin(2 * np.pi * 1.8 * t))
                phase = 2 * np.pi * np.cumsum(f0) / sr
                sig = 0.5 * np.sin(phase) + 0.3 * np.sin(2.5 * phase) + 0.15 * np.sin(4.2 * phase)
                cadence = np.sin(2 * np.pi * 3.0 * t) ** 2
                sig = sig * cadence + np.random.normal(0, 0.002, len(t))

            else:
                # Monotone / Autoregressive TTS: rigidly flat F0 std < 15Hz
                f0 = np.random.uniform(145, 185)
                phase = 2 * np.pi * f0 * t
                sig = 0.7 * np.sin(phase) + 0.3 * np.sin(2 * phase)
                cadence = np.sin(2 * np.pi * 3.8 * t) ** 2
                sig = sig * cadence + np.random.normal(0, 0.0008, len(t))

            sig = sig / (np.max(np.abs(sig)) + 1e-7) * 0.92
            out_file = os.path.join(output_dir, "ai", f"ai_neural_tts_{i+1:03d}.wav")
            wav.write(out_file, sr, (sig * 32767).astype(np.int16))

        print(f"Generated {samples_per_class*2} high-fidelity benchmark samples in {output_dir}")

if __name__ == "__main__":
    DatasetManager.generate_synthetic_benchmark_dataset("dataset", samples_per_class=60)
