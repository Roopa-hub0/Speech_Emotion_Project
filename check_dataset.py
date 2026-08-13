import numpy as np

X = np.load("cnn_lstm_X.npy")
y = np.load("cnn_lstm_y.npy")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("Classes:", np.unique(y))