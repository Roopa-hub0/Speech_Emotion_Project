from flask import Flask, render_template, request, send_file
import librosa
import librosa.display
import numpy as np
import pandas as pd
from datetime import datetime
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import joblib


os.makedirs("audio", exist_ok=True)
os.makedirs("static", exist_ok=True)
app = Flask(__name__)

attention_model = load_model("attention_cnn_model.h5")
gender_model = joblib.load("gender_model.pkl")

class_names = [
    'angry',
    'calm',
    'disgust',
    'fearful',
    'happy',
    'neutral',
    'sad',
    'surprised'
]


def create_model_spectrogram(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    temp_path = "audio/model_temp.png"

    plt.figure(figsize=(6, 6))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        cmap='viridis'
    )

    plt.axis('off')

    plt.savefig(
        temp_path,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close()

    return temp_path
def create_display_spectrogram(audio_path):
    audio, sr = librosa.load(audio_path, sr=None)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=256,
        fmax=8000
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    temp_path = "static/temp_spectrogram.png"

    plt.style.use('dark_background')

    plt.figure(
        figsize=(10, 4),
        dpi=150,
        facecolor='#1e1e2f'
    )

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        cmap='magma'
    )

    cbar = plt.colorbar(format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=8)

    plt.title("Mel Spectrogram", fontsize=14)

    plt.tight_layout()

    plt.savefig(
        temp_path,
        bbox_inches='tight',
        pad_inches=0.2,
        facecolor='#1e1e2f'
    )

    plt.close()

    return temp_path


def predict_emotion(audio_path):

    # Pretty spectrogram for UI
    create_display_spectrogram(audio_path)

    # Original spectrogram for CNN
    img_path = create_model_spectrogram(audio_path)

    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=(128, 128)
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = attention_model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions)

    emotion = class_names[predicted_index]

    confidence = round(
        np.max(predictions) * 100,
        2
    )

    return emotion, confidence, "temp_spectrogram.png"
def predict_gender(audio_path):

    audio, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_mean = mfcc_mean.reshape(1, -1)

    prediction = gender_model.predict(mfcc_mean)[0]

    if prediction == 0:
        return "Male"
    else:
        return "Female"

def get_stress_and_recommendation(emotion):

    emotion = emotion.lower()

    # Better stress mapping
    if emotion in ["happy", "calm"]:
        stress = "Low"

    elif emotion in ["neutral", "surprised"]:
        stress = "Medium"

    elif emotion == "sad":
        stress = "Medium-High"

    elif emotion in ["angry", "fearful", "disgust"]:
        stress = "High"

    else:
        stress = "Unknown"

    # Recommendations
    if emotion == "happy":
        recommendation = "Keep maintaining your positive mood."

    elif emotion == "calm":
        recommendation = "Great emotional balance detected."

    elif emotion == "neutral":
        recommendation = "Try engaging in activities that interest you."

    elif emotion == "angry":
        recommendation = "Take a short break and practice deep breathing."

    elif emotion == "fearful":
        recommendation = "Relaxation exercises may help reduce anxiety."

    elif emotion == "sad":
        recommendation = "Talk with friends or engage in enjoyable activities."

    elif emotion == "surprised":
        recommendation = "Unexpected emotion detected. Stay calm and assess the situation."

    elif emotion == "disgust":
        recommendation = "Take a moment to identify discomfort and focus on positive activities."

    else:
        recommendation = "Maintain a healthy lifestyle."

    return stress, recommendation
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

    # Prediction
    emotion, confidence, spectrogram = predict_emotion(filepath)
    gender = predict_gender(filepath)
    print("Gender:", gender)

    # Confidence check
    if confidence < 60:
        emotion = "Uncertain"
        stress = "Cannot Determine"
        recommendation = "Please provide clearer audio."

    else:
        stress, recommendation = get_stress_and_recommendation(emotion)

    # Save history
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame({
        "Time": [current_time],
        "Emotion": [emotion],
        "Gender": [gender],
        "Stress": [stress],
        "Confidence": [float(confidence)]
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
        recommendation=recommendation,
        spectrogram=spectrogram,
        gender=gender
    )


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

    # Emotion prediction
    emotion, confidence, spectrogram = predict_emotion(filepath)

    # Gender prediction
    gender = predict_gender(filepath)

    print("Gender:", gender)

    # Confidence check
    if confidence < 60:
        emotion = "Uncertain"
        stress = "Cannot Determine"
        recommendation = "Please provide clearer audio."

    else:
        stress, recommendation = get_stress_and_recommendation(emotion)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame({
        "Time": [current_time],
        "Emotion": [emotion],
        "Gender": [gender],
        "Stress": [stress],
        "Confidence": [float(confidence)]
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
        gender=gender,
        stress=stress,
        confidence=confidence,
        recommendation=recommendation,
        spectrogram=spectrogram
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


# ================= DOWNLOAD =================
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

    # ================= Emotion Chart =================
    emotion_counts = data["Emotion"].value_counts()

    plt.figure(figsize=(6, 6))

    plt.pie(
        emotion_counts,
        labels=emotion_counts.index,
        autopct='%1.1f%%'
    )

    plt.title("Emotion Distribution")
    plt.tight_layout()

    plt.savefig(
        "static/emotion_chart.png",
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close()

    # ================= Stress Chart =================
    stress_counts = data["Stress"].value_counts()

    plt.figure(figsize=(4, 3))

    bars = plt.bar(
        stress_counts.index,
        stress_counts.values,
        color=["red", "orange", "green"][:len(stress_counts)],
        width=0.35
    )

    plt.title("Stress Distribution", fontsize=14)
    plt.xlabel("Stress Level")
    plt.ylabel("Count")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.5,
            str(int(height)),
            ha='center'
        )

    plt.tight_layout()

    plt.savefig(
        "static/stress_chart.png",
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close()

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        most_common_emotion=most_common_emotion,
        high_stress_count=high_stress_count
    )
# ================= RUN APP =================
if __name__ == '__main__':
    app.run(debug=True)
    