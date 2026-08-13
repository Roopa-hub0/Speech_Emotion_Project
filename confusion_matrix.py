import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

model = tf.keras.models.load_model("augmented_spectrogram_cnn.h5")

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "augmented_spectrogram_dataset",
    image_size=(128,128),
    batch_size=32,
    shuffle=False
)

class_names = dataset.class_names

y_true = []
y_pred = []

for images, labels in dataset:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)
# ================= Classification Report =================
print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6), dpi=300)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlGnBu',
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size":8}
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

plt.xlabel("Predicted", fontsize=11)
plt.ylabel("Actual", fontsize=11)
#plt.title("Confusion Matrix", fontsize=12)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_ieee2.png",
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.05
)

plt.close()