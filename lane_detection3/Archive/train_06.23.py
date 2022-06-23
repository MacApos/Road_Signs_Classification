from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots
from sklearn.utils import shuffle
from datetime import datetime
import plotly.graph_objects as go
import plotly.offline as po
import pandas as pd
import numpy as np
import pickle
import cv2
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.callbacks import CSVLogger
from keras.optimizers import adam_v2
from keras.models import Sequential

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

np.random.seed(10)


def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='valid_accuracy',
                             mode='markers+lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='valid_loss',
                             mode='markers+lines'), row=2, col=1)

    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title='Metrics')

    po.plot(fig, filename=filename, auto_open=True)


path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
dir_path = os.path.join(path, '../Output')
data_path = os.path.join(path, 'train')
dt = datetime.now().strftime('%d.%m_%H.%M')
output_path = os.path.join(dir_path, f'initialized_{dt}')
if not os.path.exists(output_path):
    os.mkdir(output_path)

logs_path = os.path.join(output_path, 'logs.txt')
model_path = os.path.join(output_path, 'model.h5')

data = pickle.load(open('../Pickles/160x80_warp_data.p', 'rb'))
labels = pickle.load(open('../Pickles/160x80_warp_labels.p', 'rb'))

data = np.array(data)
labels = np.array(labels)

epochs = 2
learning_rate = 0.001
batch_size = 25
input_shape = data[0].shape
loss = 'mean_absolute_error'

data, labels = shuffle(data, labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)

# load check
from lane_detection3.lane_detection import im_show, visualise

for idx, image in enumerate(x_train[:2]):
    left_curve = y_train[idx][:3]
    right_curve = y_train[idx][3:]
    warp = visualise(image, left_curve, right_curve, image.shape[0]*1/3, show_lines=True)
    im_show(warp)

train_generator = ImageDataGenerator()
valid_generator = ImageDataGenerator()

train_datagen = train_generator.flow(x_train, y_train, batch_size=batch_size)
valid_datagen = valid_generator.flow(x_test, y_test, batch_size=batch_size)

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=6, activation='sigmoid'))
model.summary()

model.compile(optimizer=adam_v2.Adam(learning_rate=learning_rate),
              loss=loss,
              metrics=['accuracy'],
              run_eagerly=True)

dt = datetime.now().strftime('%d.%m_%H.%M')

csv_logger = CSVLogger(logs_path, append=True, separator=';')

history = model.fit(x=train_datagen,
                    epochs=epochs,
                    validation_data=valid_datagen,
                    callbacks=csv_logger)

logs = open(logs_path, 'a')
logs.write(f'\nepochs = {epochs}\n')
logs.write(f'batch_size = {batch_size}\n')
logs.write(f'input_shape = {input_shape}\n')
logs.write(f'loss = {loss}\n')
logs.close()
#
#     model.save(model_path)
#     plot_hist(history, output_path, logs_path)
#
