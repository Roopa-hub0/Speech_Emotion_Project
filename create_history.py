import pandas as pd

df = pd.DataFrame(columns=[
    "Time",
    "Emotion",
    "Gender",
    "Stress",
    "Confidence"
])

df.to_csv("history.csv", index=False)

print("history.csv created successfully!")