import librosa
import numpy as np

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = np.mean(
        librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        ).T,
        axis=0
    )

    # Chroma
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # Mel Spectrogram
    mel = np.mean(
        librosa.feature.melspectrogram(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # Spectral Contrast
    contrast = np.mean(
        librosa.feature.spectral_contrast(
            y=audio,
            sr=sr
        ).T,
        axis=0
    )

    # Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(audio)
    )

    # RMS Energy
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