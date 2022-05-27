from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as po
from datetime import datetime
from imutils import paths
import pandas as pd
import numpy as np
import argparse
import pickle
import cv2
import os

from keras.models import Sequential
from keras.layers import BatchNormalization, Flatten, Dense, Conv2DTranspose, Conv2D, MaxPooling2D,\
    Dropout, UpSampling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam_v2

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='valid_loss',
                             mode='markers+lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='valid_loss',
                             mode='markers+lines'), row=2, col=1)

    fig.update_xaxes(title_text='Liczba epok', row=1, col=1)
    fig.update_xaxes(title_text='Liczba epok', row=2, col=1)
    fig.update_xaxes(title_text='Accuracy', row=1, col=1)
    fig.update_xaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title='Metrics')

    po.plot(fig, filename=filename, auto_open=True)


epochs = 30
learning_rate = 0.001
batch_size = 30
input_shape = (60, 160, 3)
scale_factor = 1/8

# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
root_path = os.path.dirname(__file__)


output_path = os.path.join(path, 'output')
data_path = os.path.join(path, 'data')
pickles_path = os.path.join(root_path, 'Pickles')
labels_path = os.path.join(pickles_path, 'small_labels.p')
t_data_npy = os.path.join(pickles_path, 'data.npy')

data_list = list(paths.list_images(data_path))

if not os.path.exists(output_path):
    os.mkdir(output_path)

labels = pickle.load(open(labels_path, 'rb'))
labels = np.array(labels)
data = np.load(t_data_npy)

cv2.imshow('data', data[0])
cv2.waitKey(0)

print(len(labels))

# check
from lane_detection import params, im_show, visualise
video_list, dst = params()

for idx, image in enumerate(data):
    left_curve = labels[idx][:3]
    right_curve = labels[idx][3:]
    img = visualise(image, left_curve, right_curve, show_lines=True)
    im_show(img)


# i=0
# for video in video_list:
#     name = video['name']
#     thresh = video['thresh']
#     limit = video['limit']
#     src = video['src']
#
#     for idx, img in enumerate(data):
#         left_curve = labels[idx][:3]
#         right_curve = labels[idx][3:]
#         image = cv2.imread(data_list[idx])
#         poly, frame = visualise_perspective(image, left_curve, right_curve, src, dst, 1/8)
#         im_show(poly)
#
#     i += limit

print(f'{data.shape[0]} obrazów o rozmiarze: {data.nbytes / (1024 * 1000.0):.2f} MB')
#
# # data, labels = shuffle(data, labels)
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
#
# model = Sequential()
# # model.add(BatchNormalization(input_shape=input_shape))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape,
#                  activation='relu'))
# model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
# model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dropout(0.5))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(units=6, activation='sigmoid'))
#
# # model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
# # model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
# # model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
# # model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Flatten())
# # model.add(Dropout(0.5))
# # model.add(Dense(units=128, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(units=64, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(units=32, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(units=6, activation='sigmoid'))
#
# model.summary()
#
# datagen = ImageDataGenerator(rotation_range=10,
#                              height_shift_range=0.1,
#                              vertical_flip=True)
#
# model.compile(optimizer=adam_v2.Adam(learning_rate=learning_rate),
#               # optimizer=adam_v2.Adam(learning_rate=learning_rate),
#               loss='mean_absolute_error',
#               metrics=['accuracy'],
#               run_eagerly=True)
#
# dt = datetime.now().strftime('%d.%m_%H.%M')
# print(dt)
# model_path = os.path.join(output_path, 'model_' + dt + '.hdf5')
# checkpoint = ModelCheckpoint(filepath=model_path,
#                              monitor='val_accuracy',
#                              save_best_only=True)
#
# # history = model.fit_generator(
# #     generator=datagen.flow(x_train, y_train, batch_size=batch_size),
# #     validation_data=(x_test, y_test),
# #     steps_per_epoch=len(x_train)//batch_size,
# #     epochs=epochs,
# #     callbacks=[checkpoint])
#
# history = model.fit(
#     x=x_train,
#     y=y_train,
#     batch_size=batch_size,
#     epochs=epochs,
#     callbacks=[checkpoint],
#     validation_data=(x_test, y_test),
#     steps_per_epoch=len(x_train)//batch_size,
#     validation_steps = len(x_test)//batch_size
# )
#
# report_path = os.path.join(output_path, 'report_' + dt + '.html')
# plot_hist(history, filename=report_path)
#
# model_json = model.to_json()
# json_path = os.path.join(output_path, 'model_'+ dt +'.json')
#
# with open(json_path, 'w') as json_file:
#     json_file.write(model_json)
#
# weights_path = os.path.join(output_path, 'weights_'+ dt +'.json')
# model.save_weights(weights_path)