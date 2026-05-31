import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Image size & batch size
img_size = 224
batch_size = 32

# ✅ FULL PATHS (IMPORTANT)
train_path = "d:/plant health detection system/plant_project/plant_project/dataset/train"
val_path = "d:/plant health detection system/plant_project/plant_project/dataset/val"

# Train data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# Validation data
val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory(
    val_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20
)

# ✅ CREATE MODEL FOLDER + SAVE MODEL (FIXED)
model_dir = "d:/plant health detection system/plant_project/plant_project/model"
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "cnn_model.h5")

model.save(model_path)

print("🎉 Model training complete")
print("✅ Model saved at:", model_path)