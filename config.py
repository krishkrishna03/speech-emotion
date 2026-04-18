# Speech Emotion Recognition Configuration

## Audio Processing Settings
SAMPLE_RATE = 22050          # Hertz (Hz)
N_MFCC = 40                  # Number of MFCC coefficients
MAX_PAD_LEN = 128            # Maximum sequence length for padding

## Model Architecture
# CNN Parameters
CNN_FILTERS = [64, 128, 256]  # Filters for each conv layer
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = 2

# LSTM Parameters
LSTM_UNITS = [128, 64]       # Units for each LSTM layer
LSTM_DROPOUT = 0.2

# Dense Layers
DENSE_UNITS = [128, 64]      # Hidden units
DENSE_DROPOUT = [0.3, 0.2]   # Dropout rates

## Training Settings
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

## Dataset Settings
RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
DATA_PATH = "data"
MODEL_PATH = "models"
UPLOAD_PATH = "uploads"

## Emotion Labels
EMOTIONS = [
    'neutral',
    'calm',
    'happy',
    'sad',
    'angry',
    'fearful',
    'disgusted',
    'surprised'
]

EMOTION_EMOJI = {
    'neutral': '😐',
    'calm': '😌',
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'fearful': '😨',
    'disgusted': '🤢',
    'surprised': '😲'
}

## Flask Settings
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000
MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB

## Tips
# - Increase EPOCHS for better accuracy (slower training)
# - Increase BATCH_SIZE for faster training but less memory efficiency
# - Increase N_MFCC for more detailed features (slower)
# - Adjust LEARNING_RATE for convergence speed
# - Use GPU for faster training
