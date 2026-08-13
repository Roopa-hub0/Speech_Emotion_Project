import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

# Load Dataset
df = pd.read_csv("improved_features.csv")

print("Dataset Shape:", df.shape)

# Features
X = df.drop("emotion", axis=1).values

# Labels
y = df["emotion"]

# Encode Labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# One-Hot Encoding
y = to_categorical(y)

# Reshape for CNN
X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

print("Input Shape:", X.shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# CNN Model
model = Sequential()

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(X_train.shape[1], 1)
    )
)

model.add(
    MaxPooling1D(
        pool_size=2
    )
)

model.add(
    Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu'
    )
)

model.add(
    MaxPooling1D(
        pool_size=2
    )
)

model.add(Flatten())

model.add(
    Dense(
        256,
        activation='relu'
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        128,
        activation='relu'
    )
)

model.add(
    Dense(
        8,
        activation='softmax'
    )
)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Summary
model.summary()

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    "\nCNN Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# Save Model
model.save("cnn_emotion_model.h5")

print(
    "\nModel Saved Successfully!"
)