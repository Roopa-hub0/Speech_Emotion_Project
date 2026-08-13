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

filename = "03-01-05-01-01-01-01.wav"

emotion_code = filename.split("-")[2]

emotion = emotion_map[emotion_code]

print("Emotion Code:", emotion_code)
print("Emotion:", emotion)