import librosa
import numpy as np
import joblib

model = joblib.load("emotion_model.pkl")

file_path = "audio/live_recording.wav"

audio, sr = librosa.load(file_path, sr=None)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

mfcc_mean = np.mean(mfcc.T, axis=0)

mfcc_mean = mfcc_mean.reshape(1, -1)

prediction = model.predict(mfcc_mean)

emotion = prediction[0]

probabilities = model.predict_proba(mfcc_mean)

confidence = round(
    np.max(probabilities) * 100,
    2
)

if emotion.lower() in ["happy", "calm"]:
    stress = "Low"

elif emotion.lower() == "neutral":
    stress = "Medium"

else:
    stress = "High"

print("\nEmotion :", emotion)
print("Confidence :", confidence, "%")
print("Stress :", stress)