# Voice Detection Demo Dataset Overview

This dataset (`dataset_demo/`) contains approximately 100-140 curated short audio clips (3-5 seconds each) specifically designed to be used during your **University Presentation**.

The dataset is strictly structured to trigger the advanced **Explainable AI (XAI)** metrics we have integrated into the system, including **SHAP** feature attribution, **LIME** local boundaries, **Grad-CAM** Spectrogram approximation, and **Counterfactual** generation.

## Directory Structure

### 1. `human/` (~50 clips)
These are natural, realistic human speech samples in English, Hindi, and Telugu. 
- **Why it matters:** These files will trigger the "Human" classification. You can demonstrate the **SHAP / LIME** charts showing *positive* attribution for natural pitch variation and organic pauses (`silence_ratio > 0.1`).

### 2. `ai/` (~50 clips)
These are purely synthetic voices generated using Text-to-Speech algorithms across English, Hindi, and Telugu.
- **Why it matters:** These files will trigger the "AI Generated" classification. During your demo, use these to show your teacher the **Spectrogram (Grad-CAM)** highlighting unnatural high-frequency anomalies, and the **Counterfactual** logic explaining exactly *why* the AI flagged the clip (e.g., "To be classified as Human, pitch variance must increase").

### 3. `languages/` (~40 clips)
These clips specifically exhibit varied pronunciation, cadence, and speed across English, Hindi, and Telugu.
- **Why it matters:** This directory is perfect for demonstrating the system's robust Whisper ML language-routing capabilities. The system will accurately identify the spoken language regardless of whether it classifies the underlying acoustics as Human or AI.

---

*Note: This dataset was dynamically generated utilizing Librosa data-augmentation (pitch-shifting, time-stretching) over open-source voice anchors, alongside gTTS computational synthesis, to provide a clean, royalty-free environment for your XAI demonstrations.*
