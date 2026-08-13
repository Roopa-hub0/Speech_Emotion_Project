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

def extract_features(audio, sr):

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        ).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    mel = np.mean(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    spectral = np.mean(
        librosa.feature.spectral_contrast(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    rms = np.mean(
        librosa.feature.rms(y=audio)
    )

    return np.hstack([
        mfcc,
        chroma,
        mel,
        spectral,
        zcr,
        rms
    ])

for actor in os.listdir(dataset_path):

    actor_path = os.path.join(
        dataset_path,
        actor
    )

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

                    # Original
                    data.append(
                        list(
                            extract_features(
                                audio,
                                sr
                            )
                        ) + [emotion]
                    )

                    # Noise
                    noise_audio = (
                        audio +
                        0.005 *
                        np.random.randn(
                            len(audio)
                        )
                    )

                    data.append(
                        list(
                            extract_features(
                                noise_audio,
                                sr
                            )
                        ) + [emotion]
                    )

                    # Pitch Shift
                    pitch_audio = (
                        librosa.effects.pitch_shift(
                            audio,
                            sr=sr,
                            n_steps=2
                        )
                    )

                    data.append(
                        list(
                            extract_features(
                                pitch_audio,
                                sr
                            )
                        ) + [emotion]
                    )

                    # Time Stretch
                    stretch_audio = (
                        librosa.effects.time_stretch(
                            audio,
                            rate=0.8
                        )
                    )

                    data.append(
                        list(
                            extract_features(
                                stretch_audio,
                                sr
                            )
                        ) + [emotion]
                    )

                except Exception as e:

                    print(
                        "Error:",
                        file_path
                    )

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
    "augmented_features.csv",
    index=False
)

print("\nDataset Created Successfully!")
print("Shape:", df.shape)