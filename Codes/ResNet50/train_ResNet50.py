import matplotlib.pyplot as plt
import warnings
import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.efficientnet import EfficientNet
import pathlib
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from PIL import Image
from tensorflow.python.keras.layers import Flatten, Dense

# -----------------------
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Code' + os.path.sep + 'train_test' + os.path.sep + 'train'
sep = os.path.sep
os.chdir(OR_PATH)
# -----------------------
# -----------------------
random_seed = 42
batch_size = 64
epochs = 3
lr = 0.01
img_height = 256
img_width = 256
channel = 3
# -----------------------
# ------------------------------------------------------------------------------------------------------------------
#### Data Augmentation
# ------------------------------------------------------------------------------------------------------------------
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height, img_width, channel)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)


# ------------------------------------------------------------------------------------------------------------------
#### def
# -------------------------------------------------------------------------------------------------------------------
def save_model(model):
    '''
       receives the model and print the summary into a .txt file
  '''
    with open('model_summary_ResNet50.txt', 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


# -----------------------------------------------------------------------------------------------------------------
#### Data Load
# -----------------------------------------------------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    validation_split=0.3,
    subset="training",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    validation_split=0.3,
    subset="validation",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)


# -----------------------------------------------------------------------------------------------------------------
#### Model for Training
# -----------------------------------------------------------------------------------------------------------------
def model_def():
    model1 = tf.keras.Sequential()
    Resnet = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    for layer in Resnet.layers:
        layer.trainable = False
    model1.add(Resnet)
    model1.add(tf.keras.layers.Dense(12))

    # Add the output layer
    average_pooling = keras.layers.GlobalAveragePooling2D()(model1.output)

    output = keras.layers.Dense(12, activation='softmax')(average_pooling)

    # Get the model
    model = keras.Model(inputs=model1.input, outputs=output)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    save_model(model)
    return model


# -------------------------------------------------------------------------------------------------------------------
def train_func(train_ds):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
    check_point = tf.keras.callbacks.ModelCheckpoint('model_ResNet50.h5', monitor='val_loss', save_best_only=True, mode='min')
    model = model_def()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[early_stop, check_point])
    return history


# -----------------------------------------------------------------------------------------------------------------
### Training
# -----------------------------------------------------------------------------------------------------------------
history = train_func(train_ds)
# -----------------------
### train_val plot

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

x = np.arange(1, epochs + 1, 1)
fig = plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(x, acc, label='Training Accuracy')
plt.plot(x, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(x)
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(x, loss, label='Training Loss')
plt.plot(x, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(x)
plt.tight_layout()
fig.savefig('train_val_plot_ResNet50.pdf', bbox_inches='tight')
plt.show()
