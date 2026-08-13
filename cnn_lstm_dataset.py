import os
import librosa
import numpy as np

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

X = []
y = []

max_len = 174

for actor in os.listdir(dataset_path):

    actor_path = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                file_path = os.path.join(
                    actor_path,
                    file
                )

                emotion_code = file.split("-")[2]

                emotion = emotion_map[
                    emotion_code
                ]

                try:

                    audio, sr = librosa.load(
                        file_path,
                        sr=None
                    )

                    mfcc = librosa.feature.mfcc(
                        y=audio,
                        sr=sr,
                        n_mfcc=40
                    )

                    if mfcc.shape[1] < max_len:

                        pad_width = (
                            max_len -
                            mfcc.shape[1]
                        )

                        mfcc = np.pad(
                            mfcc,
                            pad_width=((0, 0),
                                       (0, pad_width)),
                            mode='constant'
                        )

                    else:

                        mfcc = mfcc[:, :max_len]

                    X.append(mfcc)
                    y.append(emotion)

                except Exception as e:

                    print(
                        "Error:",
                        file_path
                    )

X = np.array(X)
y = np.array(y)

print("MFCC Dataset Shape:", X.shape)
print("Labels Shape:", y.shape)

np.save("cnn_lstm_X.npy", X)
np.save("cnn_lstm_y.npy", y)

print("\nDataset Saved Successfully!")