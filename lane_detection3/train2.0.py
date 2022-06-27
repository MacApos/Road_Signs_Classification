import keras
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
from keras.optimizers import adam_v2, rmsprop_v2
from keras.models import Sequential, Model

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

    po.plot(fig, filename=os.path.join(filename, 'report.html'), auto_open=True)
    fig.write_image(os.path.join(filename, 'report.png'))

path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
dir_path = os.path.join(path, 'output')
data_path = os.path.join(path, 'train')
dt = datetime.now().strftime('%d.%m_%H.%M')
# output_path = os.path.join(dir_path, f'initialized_{dt}')
output_path = os.path.join(dir_path, f'initialized_1')
if not os.path.exists(output_path):
    os.mkdir(output_path)

logs_path = os.path.join(output_path, 'logs.txt')
model_path = os.path.join(output_path, 'model.h5')
report_path = os.path.join(output_path, 'report.html')

data = pickle.load(open('Pickles/160x80_warp_data.p', 'rb'))
labels = pickle.load(open('Pickles/160x80_warp_labels.p', 'rb'))
coefficients = pickle.load(open('Pickles/160x80_warp_coefficients.p', 'rb'))

data = np.array(data)
labels = np.array(labels)
height = data[0].shape[0]
width = data[0].shape[1]

# # load check
from lane_detection3.lane_detection import im_show, visualise
#
# y = np.linspace(0, height, 3).astype(int)
# for i, image in enumerate(data[:1]):
#     left_points = np.array(labels[0][:3] * width).astype(int)
#     right_points = np.array(labels[0][3:] * width).astype(int)
# #
#     left_curve = coefficients[i][:3]
#     right_curve = coefficients[i][3:]
#
#     warp = visualise(image, left_curve, right_curve, show_lines=True)
#     for j, y_ in enumerate(y):
#         cv2.circle(warp, (int(left_points[j]), y_), 2, (0, 255, 0), -1)
#         cv2.circle(warp, (int(right_points[j]), y_), 2, (0, 255, 0), -1)
#
#     im_show(warp)

epochs = 7
learning_rate = 0.001
batch_size = 32
input_shape = (height, width, 3)

loss = 'mean_squared_error'

shuffled_data, shuffled_labels = shuffle(data, labels)
x_train, x_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.2, random_state=10)

train_generator = ImageDataGenerator()
valid_generator = ImageDataGenerator()

train_datagen = train_generator.flow(x_train, y_train, batch_size=batch_size)
valid_datagen = valid_generator.flow(x_test, y_test, batch_size=batch_size)

# # generator check
# y_range = np.linspace(0, height, 3).astype(int)
# for i, (x, y) in enumerate(train_datagen):
#     left_points = np.array(y[0][:3] * width).astype(int)
#     right_points = np.array(y[0][3:] * width).astype(int)
#
#     index = np.where(np.all(labels == y[0], axis=1))[0][0]
#
#     left_curve = coefficients[index][:3]
#     right_curve = coefficients[index][3:]
#
#     warp = visualise(x[0], left_curve, right_curve, show_lines=True)
#
#     for j, y_ in enumerate(y_range):
#         cv2.circle(warp, (left_points[j][0], y_), 2, (0, 255, 0), -1)
#         cv2.circle(warp, (right_points[j][0], y_), 2, (0, 255, 0), -1)
#
#     im_show(warp)

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=512   , activation='relu'))
model.add(Dense(units=6, activation='linear'))
model.summary()

model.compile(loss=loss,
              optimizer=adam_v2.Adam(learning_rate=learning_rate),
              metrics=['accuracy'])

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

model.save(model_path)
plot_hist(history, filename=output_path)

