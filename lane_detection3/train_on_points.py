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

    po.plot(fig, filename=os.path.join(filename, 'report.html'), auto_open=False)
    fig.write_image(os.path.join(filename, 'report.png'))


path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
dir_path = os.path.join(path, 'output')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

warp_data = pickle.load(open('Pickles/160x80_warp_data.p', 'rb'))
warp_labels = pickle.load(open('Pickles/160x80_warp_labels.p', 'rb'))
warp_coefficients = pickle.load(open('Pickles/160x80_warp_coefficients.p', 'rb'))

data = pickle.load(open('Pickles/160x80_data.p', 'rb'))
labels = pickle.load(open('Pickles/160x80_labels.p', 'rb'))
coefficients = pickle.load(open('Pickles/160x80_coefficients.p', 'rb'))

data_type = [warp_data, data]
labels_type = [warp_labels, labels]
coefficients_type = [warp_coefficients, coefficients]
fnames = ['train_1', 'train_2']

height = data[0].shape[0]
width = data[0].shape[1]
boundaries = [0, 0.6 * height]

epochs = 30
learning_rate = 0.001
batch_size = 32
input_shape = (height, width, 3)
loss = 'mean_squared_error'

for idx in range(2):
    output_path = os.path.join(dir_path, f'{fnames[idx]}')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    logs_path = os.path.join(output_path, 'logs.txt')
    if os.path.exists(logs_path):
        os.remove(logs_path)

    model_path = os.path.join(output_path, 'model.h5')

    data = None
    data = np.array(data_type[idx])
    labels = np.array(labels_type[idx])

    coefficients = coefficients_type[idx]
    start = boundaries[idx]

    shuffled_data, shuffled_labels = shuffle(data, labels)
    x_train, x_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.2, random_state=10)

    train_generator = ImageDataGenerator()
    valid_generator = ImageDataGenerator()

    train_datagen = train_generator.flow(x_train, y_train, batch_size=batch_size)
    valid_datagen = valid_generator.flow(x_test, y_test, batch_size=batch_size)

    # generator check
    # from lane_detection3.lane_detection import im_show, visualise
    y_range = np.linspace(start, height - 1, 3).astype(int)
    # for i, (x, y) in enumerate(train_datagen):
    #     left_points = np.array(y[0][:3] * width).astype(int)
    #     right_points = np.array(y[0][3:] * width).astype(int)
    #
    #     index = np.where(np.all(labels == y[0], axis=1))[0][0]
    #
    #     left_curve = coefficients[index][:3]
    #     right_curve = coefficients[index][3:]
    #
    #     warp = visualise(x[0], left_curve, right_curve, start, show_lines=True)
    #     for j, y_ in enumerate(y_range):
    #         cv2.circle(warp, (left_points[j][0], y_), 2, (0, 255, 0), -1)
    #         cv2.circle(warp, (right_points[j][0], y_), 2, (0, 255, 0), -1)
    #     im_show(warp)
    #     if i > 6: break


    keras.backend.clear_session()

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
    model.add(Dense(units=6))
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

    predictions = model.predict(valid_datagen)
    points_arr = np.array(predictions[0] * width).astype(int).reshape((2, -1))

    mask = np.zeros((height, width))
    nonzero = []
    for arr in points_arr:
        side = np.zeros((height, width))
        for j in zip(arr, y_range):
            cv2.circle(side, (j), 4, (255, 0, 0), -1)
        mask += side

    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
