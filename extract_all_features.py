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
                    audio, sr = librosa.load(file_path, sr=None)

                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sr,
                        n_mfcc=40
                    )

                    mfcc_mean = np.mean(mfcc.T, axis=0)

                    row = list(mfcc_mean)
                    row.append(emotion)

                    data.append(row)

                except Exception as e:
                    print("Error:", file_path)

columns = [f"mfcc_{i}" for i in range(40)]
columns.append("emotion")

df = pd.DataFrame(data, columns=columns)

df.to_csv("features.csv", index=False)

print("Features Extracted Successfully!")
print("Total Samples:", len(df))
print(df.head())