import os
import numpy as np
from typing import Tuple, Dict, Any, List
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.analysis import extract_ml_features, analyze_acoustic_indicators, preprocess_audio
from training.dataset_manager import DatasetManager

def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the optimal decision threshold
    where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    Essential metric for anti-spoofing research.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    
    # EER is the intersection point where fpr == fnr
    diffs = np.abs(fpr - fnr)
    idx = np.nanargmin(diffs)
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, thresh

def train_and_export_models(data_dir: str = "dataset"):
    """
    Trains and saves:
    1. Random Forest Voice Deepfake Classifier -> app/voice_model.joblib
    2. Learned Logistic Regression Fusion Model -> app/fusion_model.joblib
    """
    print(f"=== Starting Supervised Voice Detection Model Training ===")
    print(f"Loading dataset from: {data_dir}")

    files, labels = DatasetManager.load_directory_dataset(data_dir)
    
    # If no files or very few, generate benchmark dataset
    if len(files) < 10:
        print("Dataset directory has insufficient samples. Generating synthetic and human benchmark data...")
        DatasetManager.generate_synthetic_benchmark_dataset(data_dir, samples_per_class=35)
        files, labels = DatasetManager.load_directory_dataset(data_dir)

    print(f"Total dataset size: {len(files)} audio files ({labels.count(0)} Human, {labels.count(1)} AI)")

    X_ml = []
    X_acoustic = []
    valid_labels = []

    for idx, (fpath, label) in enumerate(zip(files, labels)):
        try:
            raw_y, raw_sr = librosa.load(fpath, sr=None)
            y, sr = preprocess_audio(raw_y, raw_sr, target_sr=16000)

            # ML Features (44-dim)
            ml_feats = extract_ml_features(y, sr)
            
            # Acoustic Indicators
            _, raw_dict, _ = analyze_acoustic_indicators(y, sr)
            snr_scaled = float(np.clip(raw_dict.get("snr_db", 0.0) / 40.0, 0.0, 1.0))
            pitch_stab = float(raw_dict.get("pitch_stability", 0.5))
            silence_r = float(raw_dict.get("silence_ratio", 0.1))
            flatness = float(raw_dict.get("spectral_flatness", 0.05))

            acoustic_vec = [snr_scaled, pitch_stab, silence_r, flatness]

            X_ml.append(ml_feats)
            X_acoustic.append(acoustic_vec)
            valid_labels.append(label)
        except Exception as e:
            print(f"Warning: Failed to extract features from {fpath}: {e}")

    X_ml = np.array(X_ml)
    X_acoustic = np.array(X_acoustic)
    y_arr = np.array(valid_labels)

    # Train / Test Split (80% train, 20% test)
    X_train_ml, X_test_ml, X_train_ac, X_test_ac, y_train, y_test = train_test_split(
        X_ml, X_acoustic, y_arr, test_size=0.25, random_state=42, stratify=y_arr
    )

    print(f"Training partition: {len(y_train)} samples | Test partition: {len(y_test)} samples")

    # 1. Train Random Forest (ML Branch)
    print("\n--- Training Supervised Random Forest Classifier ---")
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, oob_score=True)
    rf_clf.fit(X_train_ml, y_train)

    rf_train_probs = rf_clf.oob_decision_function_[:, 1]
    rf_test_probs = rf_clf.predict_proba(X_test_ml)[:, 1]
    rf_preds = rf_clf.predict(X_test_ml)

    rf_acc = accuracy_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_test_probs)
    rf_eer, _ = compute_eer(y_test, rf_test_probs)

    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")
    print(f"Random Forest ROC-AUC:       {rf_auc:.4f}")
    print(f"Random Forest EER:           {rf_eer * 100:.2f}%")

    # 2. Train Learned Logistic Regression Fusion Model (Branch Fusion)
    print("\n--- Training Learned Logistic Regression Fusion Model ---")
    # Feature for fusion: [rf_prob, snr_scaled, pitch_stability, silence_ratio, flatness]
    X_train_fusion = np.column_stack([rf_train_probs, X_train_ac])
    X_test_fusion = np.column_stack([rf_test_probs, X_test_ac])

    fusion_clf = LogisticRegression(random_state=42)
    fusion_clf.fit(X_train_fusion, y_train)

    fusion_test_probs = fusion_clf.predict_proba(X_test_fusion)[:, 1]
    fusion_preds = (fusion_test_probs >= 0.5).astype(int)

    fus_acc = accuracy_score(y_test, fusion_preds)
    fus_auc = roc_auc_score(y_test, fusion_test_probs)
    fus_eer, _ = compute_eer(y_test, fusion_test_probs)

    print(f"Learned Fusion Test Accuracy: {fus_acc * 100:.2f}%")
    print(f"Learned Fusion ROC-AUC:       {fus_auc:.4f}")
    print(f"Learned Fusion EER:           {fus_eer * 100:.2f}%")
    print(f"Learned Coefficients:         {fusion_clf.coef_[0]}")

    # 3. Export trained models to app/
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app'))
    rf_model_out = os.path.join(app_dir, "voice_model.joblib")
    fusion_model_out = os.path.join(app_dir, "fusion_model.joblib")

    joblib.dump(rf_clf, rf_model_out)
    joblib.dump(fusion_clf, fusion_model_out)

    print(f"\n[OK] Random Forest Model exported to: {rf_model_out}")
    print(f"[OK] Fusion Model exported to:         {fusion_model_out}")

    print("\nClassification Report (Learned Fusion):")
    print(classification_report(y_test, fusion_preds, target_names=['Human', 'AI']))

if __name__ == "__main__":
    train_and_export_models("dataset")
