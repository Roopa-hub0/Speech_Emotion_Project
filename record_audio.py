import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample Rate
duration = 5  # seconds

print("🎤 Recording started... Speak now!")

recording = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=1,
    dtype='int16'
)

sd.wait()

write("audio/live_recording.wav", fs, recording)

print("✅ Recording saved as audio/live_recording.wav")