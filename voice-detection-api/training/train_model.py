import os
import numpy as np
import librosa
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented_data = y + noise_factor * noise
    return augmented_data

def shift_pitch(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def stretch_time(y, rate=1.1):
    return librosa.effects.time_stretch(y=y, rate=rate)

def apply_random_gain(y, low=0.7, high=1.3):
    return y * np.random.uniform(low, high)

def extract_features(file_path, augment=False, test_harder=False):
    """
    Extract features for AI vs Human voice detection.
    Recalibrated to 3s and 13 MFCCs for better human voice resolution.
    """
    try:
        # 3.0s window provides better phonetic context than 1.5s
        y, sr = librosa.load(file_path, duration=3.0, sr=16000)
        
        # 1. Strict Loudness Normalization
        y = librosa.util.normalize(y)

        # 2. Base noise for robustness
        noise_level = np.random.uniform(0.01, 0.03) if test_harder else 0.002
        base_noise = np.random.normal(0, noise_level, len(y))
        y = y + base_noise

        if augment:
            choice = np.random.choice(['noise', 'pitch', 'stretch', 'none'])
            if choice == 'noise':
                y = add_noise(y, noise_factor=np.random.uniform(0.01, 0.03))
            elif choice == 'pitch':
                y = shift_pitch(y, sr, n_steps=np.random.uniform(-2, 2))
            elif choice == 'stretch':
                y = stretch_time(y, rate=np.random.uniform(0.8, 1.2))

        # 3. Standard 13 MFCCs (captures vocal tract resonances better than 6)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # 4. Bring back key spectral features for AI artifacts
        flatness = np.array([np.mean(librosa.feature.spectral_flatness(y=y))])
        zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(y))])

        # Combine features (total 13+13+1+1 = 28 features)
        features = np.hstack([
            mfccs_mean, 
            mfccs_std,
            flatness,
            zcr
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def collect_data_paths(data_dir):
    paths = []
    labels = []
    classes = {'human': 0, 'ai': 1}
    for label_name, label_idx in classes.items():
        class_dir = os.path.join(data_dir, label_name)
        if not os.path.exists(class_dir): continue
        files = [f for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3', '.flac'))]
        for f in files:
            paths.append(os.path.join(class_dir, f))
            labels.append(label_idx)
    return np.array(paths), np.array(labels)

def train_model(data_dir):
    print(f"Collecting data from {data_dir}...")
    paths, labels = collect_data_paths(data_dir)
    if not len(paths): return

    # Split before augmentation
    X_train_paths, X_test_paths, y_train_labels, y_test_labels = train_test_split(
        paths, labels, test_size=0.45, random_state=42, stratify=labels
    )
    
    X_train = []
    y_train = []
    print("Processing training set...")
    for path, label in zip(X_train_paths, y_train_labels):
        feat = extract_features(path, augment=False)
        if feat is not None:
            X_train.append(feat)
            y_train.append(label)
            for _ in range(3): # Variants
                feat_aug = extract_features(path, augment=True)
                if feat_aug is not None:
                    X_train.append(feat_aug)
                    y_train.append(label)

    X_test = []
    y_test = []
    print("Processing test set (adding realism noise)...")
    for path, label in zip(X_test_paths, y_test_labels):
        for _ in range(3): # Test variants
            feat = extract_features(path, augment=False, test_harder=True)
            if feat is not None:
                X_test.append(feat)
                y_test.append(label)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Moderate model for better generalization
    clf = RandomForestClassifier(
        n_estimators=40,
        max_depth=5,
        min_samples_leaf=8,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*30)
    print(f"ACCURACY: {acc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human", "AI"]))

    # Confusion matrix with TP, FN, FP, TN labeling
    cm = confusion_matrix(y_test, y_pred, labels=[0,1])  # 0=Human, 1=AI
    cm_labeled = np.array([
        [f"{cm[0,0]} (TP)", f"{cm[0,1]} (FN)"],
        [f"{cm[1,0]} (FP)", f"{cm[1,1]} (TN)"]
    ])
    df_cm = pd.DataFrame(cm_labeled, index=['Actual Human', 'Actual AI'],
                         columns=['Predicted Human', 'Predicted AI'])
    
    print("\nConfusion Matrix:")
    print(df_cm)
    print("="*30)

    # Save model bundle
    os.makedirs('app', exist_ok=True)
    joblib.dump({'model': clf, 'scaler': scaler}, os.path.join('app', 'voice_model_bundle.joblib'))
    print(f"\nModel bundle saved to app/voice_model_bundle.joblib")

if __name__ == "__main__":
    DATASET_PATH = "dataset_demo"
    if not os.path.exists(DATASET_PATH): DATASET_PATH = "dataset"
    if os.path.exists(DATASET_PATH): train_model(DATASET_PATH)
    else: print(f"Error: {DATASET_PATH} not found.")
