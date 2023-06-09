# FinalProject-Group4
### Introducation

Image recognition is a popular technology, which can accurately identify the visual content in the image, and provide a variety of objects, scenes, and concept labels, with the ability of target detection and attribute recognition, to help clients accurately identify and understand the image. The technique has been applied to multiple fields such as text recognition, license plate recognition, and face recognition. The team would explore the topic of computer vision primarily, applying the image classification technique to animal detection. The animal classification could be used in animal management and tag the different animal species, for further application, it could be developed and suitable for animal photo recognition, early childhood education science, and image content analysis, to help people get a better idea of animals and the diversity of nature. The team’s objectives are to design a CNN model with 90% accuracy, which will be accomplished by exploring the architecture of the CNN model and comparing the state-of-the-art pre-trained models and the model with the customized architecture. The team is using TensorFlow to implement the network, for the reason that TensorFlow is a powerful and mature deep learning library with strong visualization capabilities, and there are multiple options for advanced model development. With the data augmentation of the training, the data will expand up the training dataset, therefore, the model can be fitted better. The hyperparameters that learning rate, number of epochs, batch size also are determined experimentally to achieve the highest accuracy. 

### Dataset Description
The team is going to use the “Animal Image Classification Dataset,” a source from Kaggle, which contains 12 classes of animals and at least 1,200 image files for each class.

## Proposal in Proposal folder

#### Dataset Download
1. dirtecly download from kaggle: https://www.kaggle.com/datasets/piyushkumar18/animal-image-classification-dataset and then unzip the dataset
2. unzip dataset: 
- unzip animalclassification.zip
3. rename the dataset:
from pathlib import Path
DATA_DIR = os.getcwd() + os.path.sep + 'animalclassifcation' + os.path.sep
path = Path("D:\test")
temp = path.rename("Data")
new_path = temp.absolute()
## Coding part in Code folder
#### library required
import sklearn
import numpy
import tensorflow.keras
import seaborn
import matplotlib
import tensorflow
import pandas
from PIL import Image
### Dataset Splitting
train_test_split.py
### EDA
EDA_dataset.py
### Data Augmentation
data_augmentation.py
### Model ResNet50
1. Training model to get model.h5, model_summary.txt and train_valid_acc_loss_plot.pdf
train_ResNet50.py
2. Testing model to load the model.h5 and obtain result_xlsx and results of list of metrics
test_ResNet50.py
### Model VGG19
1. Training model to get model.h5, model_summary.txt and train_valid_acc_loss_plot.pdf
train_VGG19.py
2. Testing model to load the model.h5 and obtain result_xlsx and results of list of metrics
test_VGG19.py
### Model customized architeture
1. Training model to get model.h5, model_summary.txt and train_valid_acc_loss_plot.pdf
train_cus.py
2. Testing model to load the model.h5 and obtain result_xlsx and results of list of metrics
test_cus.py
## Report in report folder
check out the report
