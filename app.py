"""
Flask app for Speech Emotion Recognition
Handles audio upload and emotion prediction
"""

import os
import numpy as np
import librosa
import pickle
from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from werkzeug.utils import secure_filename
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder
os.makedirs('uploads', exist_ok=True)

# Emotion labels
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
EMOTION_COLORS = {
    'neutral': '#808080',
    'calm': '#4169E1',
    'happy': '#FFD700',
    'sad': '#4169E1',
    'angry': '#FF0000',
    'fearful': '#9932CC',
    'disgusted': '#228B22',
    'surprised': '#FF69B4'
}

# Load model and encoders
model = None
label_encoder = None
feature_params = None

def load_models():
    """Load trained model and encoders"""
    global model, label_encoder, feature_params, predictor
    
    try:
        print("Loading model...")
        model = keras.models.load_model('models/emotion_model.h5')
        print("✓ Model loaded")
        
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✓ Label encoder loaded")
        
        with open('models/feature_params.pkl', 'rb') as f:
            feature_params = pickle.load(f)
        print("✓ Feature params loaded")
        
        # Initialize predictor with correct parameters
        max_len = feature_params.get('max_len', 100)
        predictor = EmotionPredictor(
            sr=feature_params['sr'],
            n_mfcc=feature_params['n_mfcc'],
            max_len=max_len
        )
        
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

class EmotionPredictor:
    """Predict emotion from audio"""
    
    def __init__(self, sr=22050, n_mfcc=40, max_len=100):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
    
    def extract_features(self, audio_path):
        """Extract audio features - Returns (time_steps, features)"""
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
            print(f"Error extracting features: {e}")
            return None
    
    def pad_features(self, features):
        """Pad features to fixed length
        Input: (time_steps, features)
        Output: (max_len, features)
        """
        if features.shape[0] < self.max_len:
            pad_width = ((0, self.max_len - features.shape[0]), (0, 0))
            features = np.pad(features, pad_width, mode='constant', constant_values=0)
        else:
            features = features[:self.max_len, :]
        return features
    
    def predict(self, audio_path):
        """Predict emotion"""
        try:
            # Extract features
            features = self.extract_features(audio_path)
            if features is None:
                return None, None
            
            # Pad to expected length
            features = self.pad_features(features)
            
            # Normalize
            features = (features - features.mean()) / (features.std() + 1e-8)
            
            # Add batch dimension: (1, max_len, features)
            features = np.expand_dims(features, axis=0)
            
            # Predict
            predictions = model.predict(features, verbose=0)
            emotion_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][emotion_idx])
            emotion = EMOTION_LABELS[emotion_idx]
            
            return emotion, confidence
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None

# Initialize predictor
predictor = EmotionPredictor(sr=22050, n_mfcc=40, max_len=100)

@app.route('/')
def index():
    """Serve the HTML interface"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle audio upload and prediction"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Check if file is provided
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict emotion
        emotion, confidence = predictor.predict(filepath)
        
        if emotion is None:
            return jsonify({'error': 'Error processing audio file'}), 500
        
        # Clean up
        os.remove(filepath)
        
        # Return prediction
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'color': EMOTION_COLORS.get(emotion, '#000000'),
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'running',
        'model_loaded': model_loaded,
        'emotions': EMOTION_LABELS
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Speech Emotion Recognition - Web App")
    print("=" * 60)
    
    # Load models
    if not load_models():
        print("\n✗ Failed to load model!")
        print("Please run: python train_model.py")
    else:
        print("\n✓ Application ready!")
        print("Open http://localhost:5000 in your browser")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5000)
