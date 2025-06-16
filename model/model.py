from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense,Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Path to dataset
dataset_path = r"C:\Users\manib\OneDrive\Desktop\Projects\dataset"

# Image parameters
img_height, img_width = 48, 48
batch_size = 32
num_classes = 6  # Ahegao, Angry, Happy, Neutral, Sad, Surprise

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Improved CNN model
model = Sequential([
    Input(shape=(img_height, img_width, 1)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary (optional for debugging)
model.summary()

# Callbacks (optional: EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
    ModelCheckpoint("model/emotion_model.keras", save_best_only=True)
]

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,  # more epochs to learn better
    callbacks=callbacks
)

# Save final model (backup)
os.makedirs("model", exist_ok=True)
model.save("model/emotion_model_final.keras")
