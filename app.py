from flask import Flask, render_template, request
import librosa
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from flask import send_file

app = Flask(__name__)

# Load Model
model = joblib.load("emotion_model.pkl")
def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    # MFCC (40)
    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        ).T,
        axis=0
    )

    # Chroma (12)
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # Mel Spectrogram (128)
    mel = np.mean(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # Spectral Contrast (7)
    contrast = np.mean(
        librosa.feature.spectral_contrast(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # Zero Crossing Rate (1)
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    # RMS Energy (1)
    rms = np.mean(
        librosa.feature.rms(y=audio)
    )

    features = np.hstack([
        mfcc,
        chroma,
        mel,
        contrast,
        zcr,
        rms
    ])

    return features


# ================= HOME =================
@app.route('/')
def home():
    return render_template('index.html')


# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['audio']

    filepath = "audio/uploaded.wav"
    file.save(filepath)

    # Extract Features
    features = extract_features(filepath)

    print("Feature Length:", len(features))

    features = features.reshape(1, -1)

    prediction = model.predict(features)
    emotion = prediction[0]

    probabilities = model.predict_proba(features)

    confidence = round(
        np.max(probabilities) * 100,
        2
    )

    # Stress Detection
    if emotion.lower() in ["happy", "calm"]:
        stress = "Low"

    elif emotion.lower() == "neutral":
        stress = "Medium"

    else:
        stress = "High"

    # Recommendations
    if emotion.lower() == "happy":
        recommendation = "Keep maintaining your positive mood."

    elif emotion.lower() == "calm":
        recommendation = "Great emotional balance detected."

    elif emotion.lower() == "neutral":
        recommendation = "Try engaging in activities that interest you."

    elif emotion.lower() == "angry":
        recommendation = "Take a short break and practice deep breathing."

    elif emotion.lower() == "fearful":
        recommendation = "Relaxation exercises may help reduce anxiety."

    elif emotion.lower() == "sad":
        recommendation = "Talk with friends or engage in enjoyable activities."

    elif emotion.lower() == "surprised":
        recommendation = "Stay calm and evaluate the situation carefully."

    elif emotion.lower() == "disgust":
        recommendation = "Take a moment to identify the cause of discomfort and focus on positive activities."

    else:
        recommendation = "Maintain a healthy lifestyle and positive mindset."

    # Save History
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame({
        "Time": [current_time],
        "Emotion": [emotion],
        "Stress": [stress],
        "Confidence": [confidence]
    })

    new_row.to_csv(
        "history.csv",
        mode="a",
        header=False,
        index=False
    )

    return render_template(
        'index.html',
        emotion=emotion,
        stress=stress,
        confidence=confidence,
        recommendation=recommendation
    )


# ================= HISTORY =================
@app.route('/history')
def history():

    data = pd.read_csv("history.csv")

    return render_template(
        'history.html',
        tables=data.values,
        columns=data.columns
    )

#======Download =#
@app.route('/download')
def download():

    return send_file(
        "history.csv",
        as_attachment=True
    )


# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():

    data = pd.read_csv("history.csv")

    total_predictions = len(data)

    most_common_emotion = data["Emotion"].mode()[0]

    high_stress_count = len(
        data[data["Stress"] == "High"]
    )

    # Emotion Chart
    emotion_counts = data["Emotion"].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(
        emotion_counts,
        labels=emotion_counts.index,
        autopct='%1.1f%%'
    )

    plt.title("Emotion Distribution")

    plt.savefig("static/emotion_chart.png")

    plt.close()
    stress_counts = data["Stress"].value_counts()

    plt.figure(figsize=(6,4))

    plt.bar(
    stress_counts.index,
    stress_counts.values
)

    plt.title("Stress Distribution")

    plt.savefig("static/stress_chart.png")

    plt.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        most_common_emotion=most_common_emotion,
        high_stress_count=high_stress_count
    )
print("Dashboard route executed")
# ================= RECORD AUDIO =================
@app.route('/record')
def record():

    fs = 44100
    duration = 5

    print("Recording...")

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='int16'
    )

    sd.wait()

    filepath = "audio/live_recording.wav"

    write(
        filepath,
        fs,
        recording
    )

    # Extract Features
    features = extract_features(filepath)

    print("Feature Length:", len(features))

    features = features.reshape(1, -1)

    prediction = model.predict(features)
    emotion = prediction[0]

    probabilities = model.predict_proba(features)

    confidence = round(
        np.max(probabilities) * 100,
        2
    )

    # Stress
    if emotion.lower() in ["happy", "calm"]:
        stress = "Low"

    elif emotion.lower() == "neutral":
        stress = "Medium"

    else:
        stress = "High"

    # Recommendation
    if emotion.lower() == "happy":
        recommendation = "Keep maintaining your positive mood."

    elif emotion.lower() == "calm":
        recommendation = "Great emotional balance detected."

    elif emotion.lower() == "neutral":
        recommendation = "Try engaging in activities that interest you."

    elif emotion.lower() == "angry":
        recommendation = "Take a short break and practice deep breathing."

    elif emotion.lower() == "fearful":
        recommendation = "Relaxation exercises may help reduce anxiety."

    elif emotion.lower() == "sad":
        recommendation = "Talk with friends or engage in enjoyable activities."

    elif emotion.lower() == "surprised":
        recommendation = "Stay calm and evaluate the situation carefully."

    elif emotion.lower() == "disgust":
        recommendation = "Take a moment to identify the cause of discomfort and focus on positive activities."

    else:
        recommendation = "Maintain a healthy lifestyle and positive mindset."

    # Save History
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame({
        "Time": [current_time],
        "Emotion": [emotion],
        "Stress": [stress],
        "Confidence": [confidence]
    })

    new_row.to_csv(
        "history.csv",
        mode="a",
        header=False,
        index=False
    )

    return render_template(
        'index.html',
        emotion=emotion,
        stress=stress,
        confidence=confidence,
        recommendation=recommendation
    )

# ================= RUN APP =================
if __name__ == '__main__':
    app.run(debug=True)