import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.callbacks import ModelCheckpoint
from Learning_test.architecture import model
from keras.optimizers import adam_v2


import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

np.random.seed(10)

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, help='path to the data')
ap.add_argument('-e', '--epochs', default=1, type=int, help='numbers of epochs')
args = vars(ap.parse_args())

# epochs = args['epochs']
epochs = 4
learning_rate = 0.001
batch_size = 16
input_shape = (80, 160, 3)


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


# images = list(paths.list_images(args['images']))
# image_paths = list(paths.list_images(r'F:\Nowy folder\10\Praca\Datasets\Test\downloads'))
# np.random.shuffle(image_paths)
# #
# # data = []
# # labels = []
# #
# # for image_path in image_paths:
# #     image = cv2.imread(image_path)
# #     image = cv2.resize(image, (input_shape[1], input_shape[0]))
# #     image = img_to_array(image)
# #     data.append(image)
# #
# #     label = image_path.split('\\')[-2].split('_')
# #     labels.append(label)

data = pickle.load(open('../../lane_detection3/Pickles/160x80_data.p', 'rb'))
labels = pickle.load(open('../../lane_detection3/Pickles/160x80_warp_labels.p', 'rb'))

data = np.array(data)
labels = np.array(labels)

# print(f'{len(data)} obrazów o rozmiarze: {data.nbytes / (1024 * 1000.0):.2f} MB')
# print(f'Kształt danych: {data.shape}')

# mlb = MultiLabelBinarizer()
# labels = mlb.fit_transform(labels)

# pickle.dump(mlb, open(r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\Learning_test\output\mlb.p', 'wb'))

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=10)

train_generator = ImageDataGenerator()
valid_generator = ImageDataGenerator()

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=input_shape,
                 activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=6, activation='softmax'))
model.summary()

model.compile(optimizer=adam_v2.Adam(learning_rate=learning_rate),
              loss='mean_absolute_error',
              metrics=['accuracy'])

dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
output = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\Learning_test\output'
filepath = os.path.join(output, 'model_'+dt+'.hdf5')
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             save_best_only=True)

# history = model.fit_generator(
#     generator = train_datagen.flow(x_train, y_train, batch_size=batch_size),
#     validation_data=(x_test, y_test),
#     steps_per_epoch = len(x_train) // batch_size,
#     epochs=epochs,
#     callbacks=[checkpoint]
# )

history = model.fit(
    x=train_generator.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=valid_generator.flow(x_test, y_test, batch_size=batch_size),
    steps_per_epoch=len(x_train) // batch_size,
    validation_steps=len(x_test) // batch_size)


filename = os.path.join(output, 'report_' + dt + '.html')
plot_hist(history, filename=filename)
