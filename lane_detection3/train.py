import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.offline as po
import matplotlib.pyplot as plt
from datetime import datetime
from imutils import paths
import pandas as pd
import numpy as np
import argparse
import pickle
import shutil
import cv2
import os

from keras.models import Sequential
from keras.layers import BatchNormalization, Flatten, Dense, Conv2DTranspose, Conv2D, MaxPooling2D,\
    Dropout, UpSampling2D, Activation
from keras.utils.image_utils import load_img
from keras.utils import img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator
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
    print(hist)

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

    po.plot(fig, filename=filename, auto_open=False)


epochs = [10]
learning_rate = 0.001
batch_size = 16
input_shape = (60, 160, 3)

# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
root_path = os.path.dirname(__file__)

dir_path = os.path.join(path, 'output')
data_path = os.path.join(path, 'train')
pickles_path = os.path.join(root_path, 'Pickles')

if not os.path.exists(dir_path):
    os.mkdir(dir_path)

for epoch in epochs:
    dt = datetime.now().strftime('%d.%m_%H.%M.%S')
    dir_path = os.path.join(path, 'output')
    output_path = os.path.join(dir_path, f'initialized_{dt}')
    config_path = os.path.join(output_path, 'config.txt')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    config = open(config_path,'w+')
    config.write(f'epochs = {epoch}\n')
    config.write(f'learning_rate = {learning_rate}\n')
    config.write(f'batch_size = {batch_size}\n')
    config.write(f'input_shape = {input_shape}\n')

    time.sleep(1)

    perspective = [['warp_', 0], ['', 2/3]]
    for name in perspective:
        prefix = f'{name[0]}'

        labels_path = os.path.join(pickles_path, f'{prefix}small_labels.p')
        data_npy = os.path.join(pickles_path, f'{prefix}data.npy')

        data_list = list(paths.list_images(data_path))

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        labels = pickle.load(open(labels_path, 'rb'))
        labels = np.array(labels)
        data = np.load(data_npy)

        # # load check
        # from lane_detection import im_show, visualise
        #
        # for idx, image in enumerate(data[:10]):
        #     left_curve = labels[idx][:3]
        #     right_curve = labels[idx][3:]
        #     print(left_curve, right_curve)
        #     warp = visualise(image, left_curve, right_curve, image.shape[0]*name[1], show_lines=True)
        #     im_show(warp)
        #
        print(f'{data.shape[0]} obrazów o rozmiarze: {data.nbytes / (1024 * 1000.0):.2f} MB')

        data, labels = shuffle(data, labels)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

        model = Sequential()
        # model.add(BatchNormalization(input_shape=input_shape))
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

        train_datagen = ImageDataGenerator(rotation_range=5,
                                           height_shift_range=0.1,
                                           horizontal_flip=True,
                                           rescale=1./255.)

        valid_datagen = ImageDataGenerator(rescale=1./255.)

        # generator check
        # img = data[0]
        # x = img.reshape((1,) + img.shape)
        # print(x.shape)
        #
        # i = 1
        # plt.figure(figsize=(16, 8))
        # for batch in train_datagen.flow(x, batch_size=1):
        #     plt.subplot(3, 4, i)
        #     plt.grid(False)
        #     imgplot = plt.imshow(array_to_img(batch[0]))
        #     i += 1
        #     if i % 13 == 0:
        #         break
        # plt.show()

        model.compile(optimizer=adam_v2.Adam(learning_rate=learning_rate),
                      loss='mean_absolute_error',
                      metrics=['accuracy'],
                      run_eagerly=True)

        dt = datetime.now().strftime('%d.%m_%H.%M')
        print(dt)

        history = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epoch,
            validation_data=valid_datagen.flow(x_test, y_test, batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            validation_steps=len(x_test) // batch_size)

        report_path = os.path.join(output_path, f'{prefix}report_' + dt + '.html')
        plot_hist(history, filename=report_path)

        model_json = model.to_json()
        json_path = os.path.join(output_path, f'{prefix}model_'+ dt +'.json')

        with open(json_path, 'w') as json_file:
            json_file.write(model_json)

        weights_path = os.path.join(output_path, f'{prefix}weights_'+ dt +'.h5')
        model.save_weights(weights_path)

        print(pd.DataFrame(history.history))

for folder in os.listdir(dir_path):
    folder = os.path.join(dir_path, folder)
    if not os.listdir(folder):
        shutil.rmtree(folder)


