import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Improved Dataset
#df = pd.read_csv("improved_features.csv")
df = pd.read_csv(
    "augmented_features.csv"
)
print("Dataset Shape:", df.shape)

# Features
X = df.drop("emotion", axis=1)

# Labels
y = df["emotion"]

print("\nEmotion Classes:")
print(y.unique())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Random Forest Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Model...")

# Train Model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("Random Forest Results")
print("===================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(
    model,
    "emotion_model.pkl"
)

print("\nModel Saved Successfully!")
print("File Saved As: emotion_model.pkl")