"""""
pip install split-folders
"""""
import os
import splitfolders
OR_PATH = os.getcwd()
os.chdir("..")  # Change to the parent directory
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

random_seed = 42
inputfolder = DATA_DIR

splitfolders.ratio(inputfolder, output='train_test', seed=random_seed, ratio=(0.9,0,0.1),group_prefix=None)
