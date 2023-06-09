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
DATA_DIR = os.getcwd() + os.path.sep + 'Code' + os.path.sep +'train_test' + os.path.sep + 'test'
sep = os.path.sep
os.chdir(OR_PATH)
# -----------------------
# -----------------------
AUTOTUNE = tf.data.AUTOTUNE
random_seed = 42
batch_size = 64
epochs = 3
lr = 0.01
img_height = 256
img_width = 256
channel = 3
# -----------------------
# ------------------------------------------------------------------------------------------------------------------
#### def
# ------------------------------------------------------------------------------------------------------------------
def all_data(file_dir):
    img_path = []
    img_id = []
    for root, sub_folders, files in os.walk(file_dir):
        for i in files:
            img_path.append(os.path.join(root, i))
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
    data = np.array([img_path, img_id, labels])
    data = data.transpose()
    path_list = list(data[:, 0])
    id_list = list(data[:, 1])
    label_list = list(data[:, 2])
    df = pd.DataFrame((path_list, id_list, label_list)).T
    df.rename(columns={0:'path', 1:"id", 2: 'target'}, inplace=True)
    return df


# -------------------------------------------------------------------------------------------------------------------
def process_target(target_type):
    class_names = np.sort(data['target'].unique())

    if target_type == 1:

        x = lambda x: tf.argmax(x == class_names).numpy()

        final_target = data['target'].apply(x)
        data['true'] = final_target

        final_target = to_categorical(list(final_target))

        xfinal = []

        for i in range(len(final_target)):
            joined_string = ",".join(str(int(e)) for e in (final_target[i]))
            xfinal.append(joined_string)
        final_target = xfinal

        data['target_class'] = final_target
    return class_names


# -------------------------------------------------------------------------------------------------------------------
def process_path(feature, target):
    '''
          feature is the path and id of the image
          target is the result
          returns the image and the target as label
    '''
    label = target
    file_path = feature
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=channel)
    img = tf.image.resize(img, [img_height, img_width])
    return img, label


# -------------------------------------------------------------------------------------------------------------------
def read_data():

    ds_inputs = np.array(DATA_DIR + sep +data['id'])
    ds_targets = np.array(data['true'])

    list_ds = tf.data.Dataset.from_tensor_slices((ds_inputs,ds_targets)) # creates a tensor from the image paths and targets

    final_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(batch_size)

    return final_ds


# -------------------------------------------------------------------------------------------------------------------
def predict_func(test_ds):
    final_model = tf.keras.models.load_model('model_ResNet50.h5')
    res = final_model.predict(test_ds)
    xres = [tf.argmax(f).numpy() for f in res]
    loss, accuracy = final_model.evaluate(test_ds)
    data['results'] = xres
    data.to_excel('results_ResNet50.xlsx', index=False)


# -----------------------------------------------------------------------------------------------------------------
def metrics_func(metrics, aggregates=[]):
    def f1_score_metric(y_true, y_pred, type):
        res = f1_score(y_true, y_pred, average=type)
        print("f1_score {}".format(type), res)
        return res

    def cohen_kappa_metric(y_true, y_pred):
        res = cohen_kappa_score(y_true, y_pred)
        print("cohen_kappa_score", res)
        return res

    def accuracy_metric(y_true, y_pred):
        res = accuracy_score(y_true, y_pred)
        print("accuracy_score", res)
        return res

    def matthews_metric(y_true, y_pred):
        res = matthews_corrcoef(y_true, y_pred)
        print('mattews_coef', res)
        return res

    def hamming_metric(y_true, y_pred):
        res = hamming_loss(y_true, y_pred)
        return res

    # For multiclass

    y_true = np.array(data['true'])
    y_pred = np.array(data['results'])

    # End of Multiclass

    xcont = 0
    xsum = 0
    xavg = 0

    for xm in metrics:
        if xm == 'f1_micro':
            # f1 score average = micro
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            # f1 score average = macro
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            # f1 score average =
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            # Cohen kappa
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            # Accuracy
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            # Matthews
            xmet = matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet =hamming_metric(y_true, y_pred)
        else:
            xmet = print('Metric does not exist')

        xsum = xsum + xmet
        xcont = xcont + 1

    if 'sum' in aggregates:
        print('Sum of Metrics : ', xsum)
    if 'avg' in aggregates and xcont > 0:
        print('Average of Metrics : ', xsum / xcont)


# -----------------------------------------------------------------------------------------------------------------
### Testing
# -----------------------------------------------------------------------------------------------------------------
data = all_data(DATA_DIR)
class_names= process_target(1)
test_ds = read_data()
predict_func(test_ds)
list_of_metrics = ['f1_micro', 'coh', 'acc']
list_of_agg = ['avg']
metrics_func(list_of_metrics, list_of_agg)
