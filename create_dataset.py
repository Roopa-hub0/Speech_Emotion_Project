import os
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

                emotion_code = file.split("-")[2]

                emotion = emotion_map[emotion_code]

                file_path = os.path.join(actor_path, file)

                data.append([file_path, emotion])

df = pd.DataFrame(data, columns=["Path", "Emotion"])

print(df.head())

print("\nTotal Samples:", len(df))

df.to_csv("emotion_dataset.csv", index=False)

print("\nCSV File Created Successfully!")