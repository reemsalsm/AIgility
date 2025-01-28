# first importing from kaggle 

import kaggle
!pip install kaggle
#upload ur kaggle token kaggle.json
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d hasyimabdillah/workoutexercises-images
!unzip workoutexercises-images.zip -d /content/dataset

# now classification code do it in bunches 

import os
import cv2
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import shutil
from datetime import datetime

from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3

height = 256
width = 256
channels = 3
batch_size = 256
img_shape = (height, width, channels)
img_size = (height, width)

DATA_DIR = '/content/dataset'

FILTERED_DATA_DIR = '/content/filtered_dataset'

specific_classes = ['push up', 'squat', 'barbell biceps curl', 'plank']

for class_name in specific_classes:
    src_dir = os.path.join(DATA_DIR, class_name)
    dest_dir = os.path.join(FILTERED_DATA_DIR, class_name)

    if os.path.exists(src_dir):
        if not os.path.exists(dest_dir):
            print(f"Copying '{class_name}' from {src_dir} to {dest_dir}...")
            shutil.copytree(src_dir, dest_dir)
        else:
            print(f"Class '{class_name}' already exists in the filtered dataset.")
    else:
        print(f"Class '{class_name}' does not exist in the dataset.")


for root, dirs, files in os.walk(FILTERED_DATA_DIR):
    print(f"Directory: {root}")
    print(f"Subdirectories: {dirs}")
    print(f"Files: {files}")
    print("-" * 50)

train_ds = tf.keras.utils.image_dataset_from_directory(
    FILTERED_DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.1,
    subset='training',
    image_size=img_size,
    shuffle=True,
    batch_size=batch_size,
    seed=127
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    FILTERED_DATA_DIR,
    labels='inferred',
    label_mode='categorical',
    validation_split=0.1,
    subset='validation',
    image_size=img_size,
    shuffle=False,
    batch_size=batch_size,
    seed=127
)

print("Classes:", train_ds.class_names)

labels = train_ds.class_names
print(f"Classes: {labels}")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.GaussianNoise(0.1),  # Adjusted stddev to a valid range (e.g., 0.1)
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))

def show_img(data):
    plt.figure(figsize=(10,10))
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.axis("off")

#Plotting the images in dataset
show_img(train_ds)

pre_trained = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape, pooling='avg')

for layer in pre_trained.layers:
    layer.trainable = False


# Define the model
x = pre_trained.output
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.1)(x)
predictions = tf.keras.layers.Dense(len(labels), activation='softmax')(x)

# Create the model
workout_model = tf.keras.models.Model(inputs=pre_trained.input, outputs=predictions)

workout_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
workout_model.summary()

early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 5,
                                        mode = 'auto',
                                        restore_best_weights = True
                                       )

history = workout_model.fit(train_ds,
                            validation_data = val_ds,
                            epochs = 20,
                            callbacks = [early_stopping_callback]
                           )

evaluate = workout_model.evaluate(val_ds)

epoch = range(len(history.history["loss"]))
plt.figure()
plt.plot(epoch, history.history['loss'], 'red', label = 'train_loss')
plt.plot(epoch, history.history['val_loss'], 'blue', label = 'val_loss')
plt.plot(epoch, history.history['accuracy'], 'orange', label = 'train_acc')
plt.plot(epoch, history.history['val_accuracy'], 'green', label = 'val_acc')
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

hist_df = pd.DataFrame(history.history)

hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# now saving code as keras 


current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
keras_file_path = f'workout_model_{current_datetime}.keras'
workout_model.save(keras_file_path)
print(f"Model saved as {keras_file_path}")

# Save the model in the TensorFlow SavedModel format
saved_model_dir = f'workout_model_{current_datetime}_savedmodel'
workout_model.export(saved_model_dir)  # Use the `export` method for SavedModel
print(f"Model exported in TensorFlow SavedModel format at {saved_model_dir}")

# Convert the TensorFlow SavedModel to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_file_path = f'workout_model_{current_datetime}.tflite'
with open(tflite_file_path, 'wb') as f:
    f.write(tflite_model)
print(f"Model converted and saved as {tflite_file_path}")



