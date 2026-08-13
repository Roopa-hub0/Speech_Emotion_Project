import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load Dataset
X = np.load("cnn_lstm_X.npy")
y = np.load("cnn_lstm_y.npy")

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# Reshape
X = np.transpose(X, (0, 2, 1))

print("Reshaped X:", X.shape)

# Encode Labels
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_encoded = to_categorical(y_encoded)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(y_encoded, axis=1)
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# CNN-LSTM Model
model = Sequential()

model.add(
    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(174, 40)
    )
)

model.add(
    MaxPooling1D(
        pool_size=2
    )
)

model.add(
    LSTM(
        64,
        return_sequences=False
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
    Dropout(0.3)
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

# Early Stopping
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# Evaluate
loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print(
    "\nCNN-LSTM Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# Save Model
model.save("cnn_lstm_model.h5")

print("\nModel Saved Successfully!")