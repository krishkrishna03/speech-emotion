"""
Train CNN + LSTM model for Speech Emotion Recognition
Using RAVDESS, TESS, and CREMA-D datasets
"""

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from pathlib import Path
import pickle
import json
import warnings

warnings.filterwarnings('ignore')

# Emotion mapping for different datasets
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgusted',
    '08': 'surprised'
}

TESS_EMOTION_MAP = {
    'neutral': 'neutral',
    'ps': 'sad',  # PS = Pleading/Sad
    'sad': 'sad',
    'happy': 'happy',
    'angry': 'angry',
    'disgust': 'disgusted',
    'fear': 'fearful',
    'surprised': 'surprised'
}

CREMA_EMOTION_MAP = {
    'A': 'angry',
    'D': 'disgusted',
    'F': 'fearful',
    'H': 'happy',
    'N': 'neutral',
    'S': 'sad',
    'X': 'surprised'
}

EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']

class AudioFeatureExtractor:
    """Extract acoustic features from audio files"""
    
    def __init__(self, sr=22050, n_mfcc=40):
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path):
        """Extract comprehensive audio features - Returns (time_steps, features)"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # MFCC - shape: (n_mfcc, time)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            
            # Spectral features - shape: (1, time)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zero_cross = librosa.feature.zero_crossing_rate(y)
            
            # Combine features: (43, time)
            features = np.vstack([mfcc, spec_cent, spec_roll, zero_cross])
            
            # TRANSPOSE to (time, features) format
            features = features.T
            
            return features
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None


def load_ravdess_dataset(data_path='data/RAVDESS'):
    """Load RAVDESS dataset"""
    X = []
    y = []
    feature_extractor = AudioFeatureExtractor()
    
    data_dir = Path(data_path)
    actor_dirs = sorted(data_dir.glob('Actor_*'))
    
    if not actor_dirs:
        print(f"✗ RAVDESS dataset not found at {data_path}")
        return X, y
    
    print(f"📂 Loading RAVDESS: Found {len(actor_dirs)} actors")
    
    for actor_dir in actor_dirs:
        audio_files = list(actor_dir.glob('*.wav'))
        
        for audio_file in audio_files:
            filename = audio_file.name
            try:
                # RAVDESS filename: 03-01-05-01-01-01-10.wav
                # Position 2 (0-indexed): emotion (01-08)
                emotion_code = filename.split('-')[2]
                emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
                
                if emotion:
                    features = feature_extractor.extract_features(str(audio_file))
                    if features is not None and features.shape[0] > 0:
                        X.append(features)
                        y.append(emotion)
            except Exception as e:
                pass
    
    print(f"✓ RAVDESS: Loaded {len(X)} samples")
    return X, y

def load_tess_dataset(data_path='data/TESS'):
    """Load TESS (Toronto Emotional Speech Set) dataset"""
    X = []
    y = []
    feature_extractor = AudioFeatureExtractor()
    
    data_dir = Path(data_path)
    
    # TESS structure: TESS Toronto Emotional Speech Set Data/*/
    tess_dirs = list(data_dir.glob('TESS Toronto Emotional Speech Set Data/*/'))
    
    if not tess_dirs:
        # Try alternative structure
        tess_dirs = list(data_dir.glob('*/'))
    
    if not tess_dirs:
        print(f"⚠️  TESS dataset not found at {data_path}")
        return X, y
    
    print(f"📂 Loading TESS: Found {len(tess_dirs)} speaker folders")
    
    for speaker_dir in tess_dirs:
        audio_files = list(speaker_dir.glob('*.wav'))
        
        for audio_file in audio_files:
            filename = audio_file.name
            try:
                # TESS filename: YAF_angry_sentence01.wav
                # Contains emotion in filename
                parts = filename.lower().split('_')
                if len(parts) >= 2:
                    emotion_str = parts[1]
                    emotion = TESS_EMOTION_MAP.get(emotion_str, None)
                    
                    if emotion:
                        features = feature_extractor.extract_features(str(audio_file))
                        if features is not None and features.shape[0] > 0:
                            X.append(features)
                            y.append(emotion)
            except Exception as e:
                pass
    
    print(f"✓ TESS: Loaded {len(X)} samples")
    return X, y

def load_crema_d_dataset(data_path='data/CREMA-D'):
    """Load CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)"""
    X = []
    y = []
    feature_extractor = AudioFeatureExtractor()
    
    data_dir = Path(data_path)
    audio_dir = data_dir / 'AudioWAV'
    
    if not audio_dir.exists():
        audio_files = list(data_dir.glob('**/*.wav'))
    else:
        audio_files = list(audio_dir.glob('*.wav'))
    
    if not audio_files:
        print(f"⚠️  CREMA-D dataset not found at {data_path}")
        return X, y
    
    print(f"📂 Loading CREMA-D: Found {len(audio_files)} audio files")
    
    for audio_file in audio_files:
        filename = audio_file.name
        try:
            # CREMA-D filename: 1001_DFA_angry_high.wav
            # Position 2: emotion (A/D/F/H/N/S/X)
            parts = filename.replace('.wav', '').split('_')
            if len(parts) >= 3:
                emotion_code = parts[2][0].upper()  # First letter of emotion
                emotion = CREMA_EMOTION_MAP.get(emotion_code, None)
                
                if emotion:
                    features = feature_extractor.extract_features(str(audio_file))
                    if features is not None and features.shape[0] > 0:
                        X.append(features)
                        y.append(emotion)
        except Exception as e:
            pass
    
    print(f"✓ CREMA-D: Loaded {len(X)} samples")
    return X, y

def pad_sequences(sequences, max_len=None):
    """Pad sequences to same length - expects (time_steps, features)"""
    if max_len is None:
        max_len = max(seq.shape[0] for seq in sequences)
    
    padded = []
    for seq in sequences:
        # seq shape: (time_steps, 43)
        if seq.shape[0] < max_len:
            # Pad on time axis (axis 0)
            pad_width = ((0, max_len - seq.shape[0]), (0, 0))
            seq = np.pad(seq, pad_width, mode='constant', constant_values=0)
        else:
            # Truncate if too long
            seq = seq[:max_len, :]
        padded.append(seq)
    
    return np.array(padded)  # Returns (batch, time_steps, features)


def build_cnn_lstm_model(input_shape, num_classes):
    """Build CNN + LSTM model for emotion recognition"""
    model = models.Sequential([
        # CNN layers
        layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        # LSTM layers
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model():
    """Main training function"""
    print("=" * 60)
    print("Speech Emotion Recognition - Model Training")
    print("=" * 60)
    
    # Load all available datasets
    print("\n📂 Loading datasets...")
    X_ravdess, y_ravdess = load_ravdess_dataset('data/RAVDESS')
    X_tess, y_tess = load_tess_dataset('data/TESS')
    X_crema, y_crema = load_crema_d_dataset('data/CREMA-D')
    
    # Combine all datasets
    X = X_ravdess + X_tess + X_crema
    y = y_ravdess + y_tess + y_crema
    
    if len(X) == 0:
        print("✗ No data found. Please run download_dataset.py first")
        return
    
    print(f"\n✓ Total samples loaded: {len(X)}")
    print(f"   RAVDESS: {len(X_ravdess)} | TESS: {len(X_tess)} | CREMA-D: {len(X_crema)}")
    
    # Pad sequences - IMPORTANT: max_len=100 for consistency
    print("\n📊 Preparing data...")
    X = pad_sequences(X, max_len=100)
    print(f"✓ Features shape: {X.shape}  (batch, time_steps, features)")
    
    # Encode labels
    label_encoder = {emotion: idx for idx, emotion in enumerate(EMOTION_LABELS)}
    y_encoded = np.array([label_encoder[emotion] for emotion in y])
    
    # Normalize features
    X = X.astype('float32')
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    # Split dataset
    print("\n🔀 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Build model
    print("\n🏗️ Building CNN + LSTM model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_model(input_shape, len(EMOTION_LABELS))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train model
    print("\n🚀 Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    print("\n📈 Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✓ Test Accuracy: {test_acc:.4f}")
    print(f"✓ Test Loss: {test_loss:.4f}")
    
    # Predictions
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=EMOTION_LABELS))
    
    # Save model
    print("\n💾 Saving model...")
    model.save('models/emotion_model.h5')
    
    # Save label encoder
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save feature extractor params
    with open('models/feature_params.pkl', 'wb') as f:
        pickle.dump({'sr': 22050, 'n_mfcc': 40, 'max_len': 100}, f)
    
    # Save training info
    training_info = {
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'total_samples': len(X),
        'ravdess_samples': len(X_ravdess),
        'tess_samples': len(X_tess),
        'crema_d_samples': len(X_crema),
        'emotions': EMOTION_LABELS,
        'input_shape': input_shape
    }
    
    with open('models/training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("✓ Model saved to models/emotion_model.h5")
    print("✓ Label encoder saved to models/label_encoder.pkl")
    print("\n✓ Training complete! Ready to use the app.")

if __name__ == "__main__":
    train_model()