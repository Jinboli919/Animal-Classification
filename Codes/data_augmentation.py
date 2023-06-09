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
# DATA_DIR = OR_PATH + os.path.sep + 'Deep-Learning/animalclassification/Code/train_test/train/'

DATA_DIR = os.getcwd() + os.path.sep + 'Code' + os.path.sep +'train_test' + os.path.sep + 'train'
sep = os.path.sep
os.chdir(OR_PATH)
# -----------------------
# -----------------------
random_seed = 42
train_size = 0.8

batch_size = 64
epochs = 3
lr = 0.01
img_height = 224
img_width = 224
channel = 3
# -----------------------
# -----------------------------------------------------------------------------------------------------------------
#### Data Load
# -----------------------------------------------------------------------------------------------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    validation_split=0.1,
    subset="training",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    validation_split=0.1,
    subset="validation",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)

# -----------------------------------------------------------------------------------------------------------------
#### Image plot
# -----------------------------------------------------------------------------------------------------------------
def visualize(original, augmented):
  fig = plt.figure()
  plt.subplot(1,2,1)
  plt.title('Original image')
  plt.imshow(original)
  plt.subplot(1,2,2)
  plt.title('Augmented image')
  plt.imshow(augmented)
  plt.tight_layout()

def flip(data):
    data_flipped = tf.image.flip_left_right(data)
    return data_flipped
def rotation(data):
    data_rotated = tf.image.rot90(data)
    return data_rotated

input_size = [img_height, img_width]
def resize(data):
    data_resized = tf.image.resize(data, input_size)
    return data_resized

for images, labels in train_ds.take(1):
    img = images[1].numpy().astype("uint8")
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(class_names[labels[1]], fontdict={'fontsize':12})
    plt.axis("off")
    plt.show()
    plt.figure(figsize=(12, 8))
    visualize(img, flip(img))
    plt.show()
    plt.figure(figsize=(12, 8))
    visualize(img, rotation(img))
    plt.show()

