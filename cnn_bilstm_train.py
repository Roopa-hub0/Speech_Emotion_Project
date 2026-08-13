import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ================= LOAD DATA =================
X = np.load("cnn_lstm_X.npy")
y = np.load("cnn_lstm_y.npy")

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)


# ================= TRANSPOSE =================
# (samples, 40, 174) -> (samples, 174, 40)
X = np.transpose(X, (0, 2, 1))

print("Transposed X shape:", X.shape)


# ================= LABEL ENCODING =================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)


# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# ================= MODEL =================
model = Sequential([

    Conv1D(
        filters=64,
        kernel_size=3,
        activation='relu',
        input_shape=(174, 40)
    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation='relu'
    ),

    BatchNormalization(),

    MaxPooling1D(pool_size=2),

    Bidirectional(
        LSTM(
            128,
            return_sequences=False
        )
    ),

    Dropout(0.4),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        8,
        activation='softmax'
    )
])


# ================= COMPILE =================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ================= EARLY STOPPING =================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True
)


# ================= TRAIN =================
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop]
)


# ================= EVALUATE =================
loss, accuracy = model.evaluate(X_test, y_test)

print("\nCNN + BiLSTM Accuracy:", round(accuracy * 100, 2), "%")


# ================= SAVE MODEL =================
model.save("cnn_bilstm_model.h5")

print("\nModel Saved Successfully!")