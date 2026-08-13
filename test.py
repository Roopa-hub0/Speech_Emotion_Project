import tensorflow as tf

dataset_path = "augmented_spectrogram_dataset"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128,128),
    batch_size=32
)

print("\nClass Names:")
print(train_ds.class_names)