import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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
output_path = "augmented_spectrogram_dataset"

os.makedirs(output_path, exist_ok=True)

for emotion in emotion_map.values():
    os.makedirs(
        os.path.join(output_path, emotion),
        exist_ok=True
    )

count = 0

def save_spectrogram(audio, sr, save_path):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    plt.figure(figsize=(6, 6))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        cmap='viridis'
    )

    plt.axis('off')

    plt.savefig(
        save_path,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close()

for actor in os.listdir(dataset_path):

    actor_path = os.path.join(
        dataset_path,
        actor
    )

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                try:

                    file_path = os.path.join(
                        actor_path,
                        file
                    )

                    emotion_code = file.split("-")[2]

                    emotion = emotion_map[
                        emotion_code
                    ]

                    audio, sr = librosa.load(
                        file_path,
                        sr=None
                    )

                    base_name = os.path.splitext(
                        file
                    )[0]

                    # Original
                    save_spectrogram(
                        audio,
                        sr,
                        os.path.join(
                            output_path,
                            emotion,
                            base_name + "_orig.png"
                        )
                    )

                    # Noise
                    noise_audio = (
                        audio +
                        0.005 *
                        np.random.randn(
                            len(audio)
                        )
                    )

                    save_spectrogram(
                        noise_audio,
                        sr,
                        os.path.join(
                            output_path,
                            emotion,
                            base_name + "_noise.png"
                        )
                    )

                    # Pitch Shift
                    pitch_audio = (
                        librosa.effects.pitch_shift(
                            audio,
                            sr=sr,
                            n_steps=2
                        )
                    )

                    save_spectrogram(
                        pitch_audio,
                        sr,
                        os.path.join(
                            output_path,
                            emotion,
                            base_name + "_pitch.png"
                        )
                    )

                    # Time Stretch
                    stretch_audio = (
                        librosa.effects.time_stretch(
                            audio,
                            rate=0.8
                        )
                    )

                    save_spectrogram(
                        stretch_audio,
                        sr,
                        os.path.join(
                            output_path,
                            emotion,
                            base_name + "_stretch.png"
                        )
                    )

                    count += 4

                except Exception as e:

                    print(
                        "Error:",
                        file_path
                    )

                    print(e)

print(
    "\nTotal Spectrograms Created:",
    count
)