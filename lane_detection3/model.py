import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datetime import datetime
from imutils import paths
import pandas as pd
import numpy as np
import argparse
import pickle
import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from Learning_test.architecture import model

np.random.seed(10)
labels = pickle.load(open( "Pickles/lane_labels.p", "rb" ))
image_paths = list(paths.list_images(r'F:\Nowy folder\10\Praca\Datasets\Video_data\frames'))
np.random.shuffle(image_paths, labels)

image = cv2.imread(image_paths[0])
height = image.shape[0]
width = image.shape[1]

batch_size = 150
epochs = 20
learning_rate = 0.001
input_shape = (height//8, width//8, 3)

data = []
label = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
