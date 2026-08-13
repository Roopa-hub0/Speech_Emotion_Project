import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# Dataset Path
dataset_path = "augmented_spectrogram_dataset"

# Load Dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

print("\nClasses:")
print(train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ================= MODEL =================
inputs = Input(shape=(128, 128, 3))

x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D(2,2)(x)

x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D(2,2)(x)

x = Conv2D(128, (3,3), activation='relu')(x)

# ================= SE ATTENTION BLOCK =================
se = GlobalAveragePooling2D()(x)

se = Dense(32, activation='relu')(se)

se = Dense(128, activation='sigmoid')(se)

se = Reshape((1,1,128))(se)

x = Multiply()([x, se])

# ================= CONTINUE CNN =================
x = MaxPooling2D(2,2)(x)

x = Flatten()(x)

x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

outputs = Dense(8, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Early Stopping
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stop]
)

# Evaluate
loss, accuracy = model.evaluate(val_ds)

print(
    "\nAttention CNN Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# Save
model.save("attention_cnn_model.h5")

print("\nModel Saved Successfully!")