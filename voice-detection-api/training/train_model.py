import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def extract_features(file_path):
    """
    Extract features for AI vs Human voice detection.
    """
    try:
        y, sr = librosa.load(file_path, duration=30)
        
        # 1. MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # 2. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = np.mean(flatness)
        
        # 3. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        
        # 4. Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        # Combine all features into one vector
        features = np.hstack([
            mfccs_mean, 
            mfccs_std, 
            flatness_mean, 
            zcr_mean, 
            contrast_mean
        ])
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def train_model(data_dir):
    """
    Trains a Random Forest model on audio data.
    Expected directory structure:
    data_dir/
        human/
            *.wav
        ai/
            *.wav
    """
    X = []
    y = []
    
    classes = {'human': 0, 'ai': 1}
    
    for label, class_idx in classes.items():
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found.")
            continue
            
        print(f"Processing class: {label}...")
        for filename in os.listdir(class_dir):
            if filename.endswith(('.wav', '.mp3', '.flac')):
                file_path = os.path.join(class_dir, filename)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(class_idx)
    
    if not X:
        print("No data found to train on!")
        return
        
    X = np.array(X)
    y = np.array(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # Save the model
    model_path = os.path.join('app', 'voice_model.joblib')
    joblib.dump(clf, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    # You can change this to your dataset directory
    DATASET_PATH = "dataset" 
    if not os.path.exists(DATASET_PATH):
        os.makedirs(os.path.join(DATASET_PATH, "human"), exist_ok=True)
        os.makedirs(os.path.join(DATASET_PATH, "ai"), exist_ok=True)
        print(f"Created {DATASET_PATH} directory. Please place your .wav files in 'human' and 'ai' folders.")
    else:
        train_model(DATASET_PATH)
