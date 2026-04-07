# Voice Detection Dataset Overview

This dataset contains the core audio samples used to evaluate and test the Voice Detection Machine Learning system. There are **10 total samples** categorized into Human, AI-Generated, and Specific Language recordings.

## 1. Human Audio Samples (Real Voices)
These samples represent natural human speech, demonstrating natural pitch variance, standard conversational pauses, and dynamic energy fluctuations.

1. `dataset/human/test.wav` - A standard natural human baseline.
2. `dataset/languages/en_test.wav` - A conversational English human speech baseline.

## 2. AI-Generated Synthetic Samples
These samples were generated using synthetic architectures (oscillators or text-to-speech AI). They exhibit the exact mathematical anomalies our Explainable AI (XAI) detects (such as robotic pitch stability, zero natural pauses, and extreme spectral flatness).

3. `dataset/ai/synthetic_0.wav` - Baseline Synthetic Test 0
4. `dataset/ai/synthetic_1.wav` - Baseline Synthetic Test 1 
5. `dataset/ai/synthetic_2.wav` - Baseline Synthetic Test 2
6. `dataset/ai/synthetic_ai_0.wav` - Advanced AI Voice Clone 0
7. `dataset/ai/synthetic_ai_1.wav` - Advanced AI Voice Clone 1
8. `dataset/ai/synthetic_ai_2.wav` - Advanced AI Voice Clone 2
9. `dataset/ai/synthetic_ai_3.wav` - Advanced AI Voice Clone 3
10. `dataset/ai/synthetic_ai_4.wav` - Advanced AI Voice Clone 4

### How We Use Them:
During a demonstration, you can upload `human/test.wav` to show the system responding with **Human Voice** and generating SHAP/LIME charts proving natural cadence. Then, you can upload `ai/synthetic_0.wav` to show the system correctly flagging the audio as **AI-Generated**, with the Spectrogram and Attention Heatmaps highlighting the exact moments the synthetic anomalies were detected!
