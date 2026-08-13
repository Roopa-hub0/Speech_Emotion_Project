import os
import librosa
import numpy as np

dataset_path = "dataset/archive2"

X = []
y = []

for actor_folder in os.listdir(dataset_path):

    if not actor_folder.startswith("Actor_"):
        continue

    actor_path = os.path.join(dataset_path, actor_folder)

    actor_num = int(actor_folder.replace("Actor_", ""))

    if actor_num % 2 == 0:
        gender = 1   # Female
    else:
        gender = 0   # Male

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):
            filepath = os.path.join(actor_path, file)

            audio, sr = librosa.load(filepath, sr=None)

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=40
            )

            mfcc_mean = np.mean(mfcc.T, axis=0)

            X.append(mfcc_mean)
            y.append(gender)

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

np.save("gender_X.npy", X)
np.save("gender_y.npy", y)

print("Gender dataset saved!")