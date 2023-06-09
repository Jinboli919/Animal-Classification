import matplotlib.pyplot as plt
import warnings
import numpy as np
import os
import PIL
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
import pathlib
import warnings
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

# -----------------------
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
# DATA_DIR = OR_PATH + os.path.sep + 'Deep-Learning/animalclassification/Data'
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
data_dir_test = os.getcwd() + os.path.sep + 'Code' + os.path.sep +'train_test' + os.path.sep + 'test'
data_dir_train = os.getcwd() + os.path.sep + 'Code' + os.path.sep +'train_test' + os.path.sep + 'train'
sep = os.path.sep
os.chdir(OR_PATH)

# -----------------------
## Process images in parallel
AUTOTUNE = tf.data.AUTOTUNE
random_seed = 42
batch_size = 64
epochs = 2
lr = 0.01
img_height = 256
img_width = 256
channel = 3
# -----------------------
# ------------------------------------------------------------------------------------------------------------------
#### def
# ------------------------------------------------------------------------------------------------------------------
def show_data(test_ds):
    img_path = []
    img_id = []
    for i in test_ds.file_paths:
        img_path.append(i)
    for i in img_path:
        a = i.split('/')[-2:]
        img_id.append(a[0] + '/' + a[1])
    labels = []
    for j in img_path:
        class_name = j.split('/')[-2]
        if class_name == 'panda':
            labels.append('panda')
        elif class_name == 'cow':
            labels.append('cow')
        elif class_name == 'spider':
            labels.append('spider')
        elif class_name == 'butterfly':
            labels.append('butterfly')
        elif class_name == 'hen':
            labels.append('hen')
        elif class_name == 'sheep':
            labels.append('sheep')
        elif class_name == 'squirrel':
            labels.append('squirrel')
        elif class_name == 'elephant':
            labels.append('elephant')
        elif class_name == 'monkey':
            labels.append('monkey')
        elif class_name == 'cats':
            labels.append('cats')
        elif class_name == 'horse':
            labels.append('horse')
        elif class_name == 'dogs':
            labels.append('dogs')
    data = np.array([img_id, labels])
    data = data.transpose()
    image_list = list(data[:, 0])
    label_list = list(data[:, 1])
    df = pd.DataFrame((image_list, label_list)).T
    df.rename(columns={0: "id", 1: 'target'}, inplace=True)
    return df

# -------------------------------------------------------------------------------------------------------------------
### Data Loading
# -------------------------------------------------------------------------------------------------------------------
all_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DATA_DIR,
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir_test,
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir_train,
    validation_split=0.3,
    subset="training",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=data_dir_train,
    validation_split=0.3,
    subset="validation",
    seed=random_seed,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
num_classes = len(class_names)
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
#### EDA
# All dataset distribution
all_data = show_data(all_ds)
Target_all = all_data.groupby(['target']).size()
plt.figure(figsize=(16,8))
Target_all.plot.barh(fontsize=20)
plt.ylabel('Animal Class', fontsize=20)
plt.xlabel('Sample counts', fontsize=20)
plt.title("Sample Distribution by Animal Class", fontsize=20)
plt.show()
plt.close()

#Train dataset distribution
train_data = show_data(train_ds)
Target_train = train_data.groupby(['target']).size()
plt.figure(figsize=(16,8))
Target_train.plot.barh(fontsize=20)
plt.ylabel('Animal Class',fontsize=20)
plt.xlabel('Sample counts',fontsize=20)
plt.title("Train Dataset Sample Distribution by Animal Class", fontsize=20)
plt.show()
plt.close()

#validation dataset distribution
val_data = show_data(val_ds)
Target_val = val_data.groupby(['target']).size()
plt.figure(figsize=(16,8))
Target_val.plot.barh(fontsize=20)
plt.ylabel('Animal Class',fontsize=20)
plt.xlabel('Sample counts',fontsize=20)
plt.title("Validation Dataset Sample Distribution by Animal Class", fontsize=20)
plt.show()
plt.close()
#### EDA

#Test dataset distribution
test_data = show_data(test_ds)
Target_test = test_data.groupby(['target']).size()
plt.figure(figsize=(16,8))
Target_test.plot.barh(fontsize=20)
plt.ylabel('Animal Class', fontsize=20)
plt.xlabel('Sample counts', fontsize=20)
plt.title("Test Dataset Sample Distribution by Animal Class", fontsize=20)
plt.show()
plt.close()
