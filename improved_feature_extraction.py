import os
import librosa
import numpy as np
import pandas as pd

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

dataset_path = "dataset/archive2"

data = []

for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                file_path = os.path.join(actor_path, file)

                emotion_code = file.split("-")[2]
                emotion = emotion_map[emotion_code]

                try:

                    audio, sr = librosa.load(
                        file_path,
                        sr=None
                    )

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
                    spectral = np.mean(
                        librosa.feature.spectral_contrast(
                            y=audio,
                            sr=sr
                        ).T,
                        axis=0
                    )

                    # Zero Crossing Rate (1)
                    zcr = np.mean(
                        librosa.feature.zero_crossing_rate(
                            audio
                        ).T,
                        axis=0
                    )

                    # Spectral Centroid (1)
                    centroid = np.mean(
                        librosa.feature.spectral_centroid(
                            y=audio,
                            sr=sr
                        ).T,
                        axis=0
                    )

                    # Spectral Rolloff (1)
                    rolloff = np.mean(
                        librosa.feature.spectral_rolloff(
                            y=audio,
                            sr=sr
                        ).T,
                        axis=0
                    )

                    # RMS Energy (1)
                    rms = np.mean(
                        librosa.feature.rms(
                            y=audio
                        ).T,
                        axis=0
                    )

                    features = np.hstack([
                        mfcc,
                        chroma,
                        mel,
                        spectral,
                        zcr,
                        centroid,
                        rolloff,
                        rms
                    ])

                    row = list(features)
                    row.append(emotion)

                    data.append(row)

                except Exception as e:

                    print(
                        "Error:",
                        file_path
                    )
                    print(e)

feature_count = len(data[0]) - 1

columns = [
    f"feature_{i}"
    for i in range(feature_count)
]

columns.append("emotion")

df = pd.DataFrame(
    data,
    columns=columns
)

df.to_csv(
    "improved_features.csv",
    index=False
)

print("\nFeature Extraction Completed!")
print("Total Samples:", len(df))
print("Total Features:", feature_count)
print("Dataset Shape:", df.shape)