import librosa
import numpy as np
import joblib

# Load saved model
model = joblib.load("emotion_model.pkl")

# Give path of audio file
file_path = r"dataset/archive2/Actor_01/03-01-05-01-01-01-01.wav"

# Extract MFCC
audio, sr = librosa.load(file_path, sr=None)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

mfcc_mean = np.mean(mfcc.T, axis=0)

# Reshape for prediction
mfcc_mean = mfcc_mean.reshape(1, -1)

# Predict
prediction = model.predict(mfcc_mean)

print("Predicted Emotion:", prediction[0])