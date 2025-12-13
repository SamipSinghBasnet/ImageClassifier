import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------------------------
# 1️⃣ SAFE ABSOLUTE PATH (FIXES 1292 IMAGE ISSUE)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")

print("Loading data from:", DATA_DIR)

# -------------------------------------------------
# 2️⃣ LOAD DATASET (48x48 GRAYSCALE)
# -------------------------------------------------
data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=(48, 48),
    batch_size=32,
    color_mode='grayscale',
    shuffle=True
)

class_names = data.class_names
print("Class names:", class_names)

# -------------------------------------------------
# 3️⃣ NORMALIZE (0–1)
# -------------------------------------------------
data = data.map(lambda x, y: (x / 255.0, y))

# -------------------------------------------------
# 4️⃣ DATASET SPLIT
# -------------------------------------------------
dataset_size = len(data)
train_size = int(dataset_size * 0.7)
val_size   = int(dataset_size * 0.2)

train = data.take(train_size)
val   = data.skip(train_size).take(val_size)
test  = data.skip(train_size + val_size)

print('tota images approx:', dataset_size)

print("Train batches:", len(train))
print("Val batches:", len(val))
print("Test batches:", len(test))
print("Approx total images:", len(data) * 32)

# -------------------------------------------------
# 5️⃣ CNN MODEL (EMOTION OPTIMIZED)
# -------------------------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 emotions
])

model.summary()

# -------------------------------------------------
# 6️⃣ COMPILE
# -------------------------------------------------
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# -------------------------------------------------
# 7️⃣ EARLY STOPPING (PREVENT OVERFITTING)
# -------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -------------------------------------------------
# 8️⃣ TRAIN
# -------------------------------------------------
history = model.fit(
    train,
    validation_data=val,
    epochs=30,
    callbacks=[early_stop]
)

# -------------------------------------------------
# 9️⃣ EVALUATE
# -------------------------------------------------
test_loss, test_acc = model.evaluate(test)
print(f"Test Accuracy: {test_acc:.4f}")

# -------------------------------------------------
# 🔟 SAVE MODEL
# -------------------------------------------------
model.save("emotion_model_gray48.h5")
print("Model saved as emotion_model_gray48.h5")

# -------------------------------------------------
# 1️⃣1️⃣ PLOT TRAINING CURVES
# -------------------------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()
