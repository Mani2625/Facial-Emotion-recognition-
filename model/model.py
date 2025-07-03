import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np

# Paths
train_path = r"Downloads\emotion dataset\train"
test_path = r"Downloads\emotion dataset\test"
img_height, img_width = 224, 224
batch_size = 32
num_classes = 7

# Data generators

# Improved data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)


# Load EfficientNetB0
base_model = EfficientNetB0(include_top=False, input_shape=(img_height, img_width, 3), weights='imagenet')
base_model.trainable = False  # freeze initially

# Improved head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Use AdamW optimizer for better generalization
try:
    from tensorflow.keras.optimizers import AdamW
    optimizer = AdamW(learning_rate=1e-4)
except ImportError:
    optimizer = Adam(1e-4)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# Callbacks
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(patience=4, factor=0.3, verbose=1, min_lr=1e-6),
    ModelCheckpoint("model/emotion_model_efficientnet.keras", save_best_only=True, monitor='val_accuracy', mode='max')
]


# Train
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    callbacks=callbacks
)


# Fine-tune EfficientNetB0 (unfreeze top 60 layers)
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

# Use lower learning rate for fine-tuning
try:
    optimizer_finetune = AdamW(learning_rate=5e-6)
except:
    optimizer_finetune = Adam(5e-6)

model.compile(optimizer=optimizer_finetune, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"\n✅ Final Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Save final model
os.makedirs("model", exist_ok=True)
model.save("model/emotion_model_efficientnet.keras")
