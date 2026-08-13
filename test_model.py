from tensorflow.keras.models import load_model

model = load_model(
    "augmented_spectrogram_cnn.h5"
)

print("Loaded Successfully")

print(model.input_shape)

print(model.output_shape)