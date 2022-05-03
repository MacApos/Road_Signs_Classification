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
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Deconvolution2D, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam



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

    po.plot(fig, filename=filename, auto_open=False)


epochs = 20
learning_rate = 0.001
batch_size = 16
input_shape = (60, 160, 3)

data_path = list(paths.list_images(r'F:\Nowy folder\10\Praca\Datasets\Video_data\video2'))
file = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\data.npy'
labels = pickle.load(open( r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3'
                           r'\Pickles\lane_labels_2.p', 'rb' ))

output = r'F:\Nowy folder\10\Praca\Datasets\Video_data\output'


if os.path.exists(file):
    data = np.load(file)
else:
    data = []
    for path in data_path:
        image = cv2.imread(path)
        image = cv2.resize(image, (input_shape[1], input_shape[0]))
        image = img_to_array(image)
        data.append(image)

    data = np.array(data, dtype='float') / 255.
    np.save(file, data)

labels = np.array(labels)

print(f'{len(data_path)} obrazów o rozmiarze: {data.nbytes / (1024 * 1000.0):.2f} MB')

data, labels = shuffle(data, labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print(x_train.shape)
print(x_test.shape)

# model = Sequential()
# model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=input_shape,
#                  activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(units=128, activation='relu'))
# model.add(Dense(units=84, activation='relu'))
# model.add(Dense(units=1, activation='sigmoid'))

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))

model.add(Conv2D(filters=60, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
model.add(Conv2D(filters=50, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=40, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(filters=30, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(filters=20, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=10, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(filters=5, kernel_size=(3, 3), strides=(1,1), padding='valid', activation='relu'))
model.add(Dropout(0.2))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(UpSampling2D(size=(2, 2)))

model.add(Deconvolution2D(filters=10, kernel_size=(3, 3), strides=(1,1), padding='valid',  activation = 'relu'))
model.add(Dropout(0.2))

model.add(Deconvolution2D(filters=20, kernel_size=(3, 3), strides=(1,1), padding='valid',  activation = 'relu'))
model.add(Dropout(0.2))

model.add(UpSampling2D(size=(2, 2)))

model.add(Deconvolution2D(filters=30, kernel_size=(3, 3), strides=(1,1), padding='valid',  activation = 'relu'))
model.add(Dropout(0.2))

model.add(Deconvolution2D(filters=40, kernel_size=(3, 3), strides=(1,1), padding='valid',  activation = 'relu'))
model.add(Dropout(0.2))

model.add(Deconvolution2D(filters=50, kernel_size=(3, 3), strides=(1,1), padding='valid',  activation = 'relu'))
model.add(Dropout(0.2))

model.add(UpSampling2D(size=(2, 2)))

model.add(Deconvolution2D(filters=60, kernel_size=(3, 3), strides=(1,1), padding='valid',  activation = 'relu'))
model.add(Dropout(0.2))

model.add(Deconvolution2D(filters=1, kernel_size=(3, 3), strides=(1,1), padding='valid',  activation = 'relu'))
model.add(Dropout(0.2))

model.summary()

datagen = ImageDataGenerator()

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss = 'mean_squared_error',
              metrics=['accuracy'])

dt = datetime.now().strftime('%d/%m_%H:%M')
filepath = os.path.join(output, 'model_'+dt+'.hdf5')
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             save_best_only=True)

history = model.fit_generator(
    generator = datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test, y_test),
    steps_per_epoch = len(x_train) // batch_size,
    epochs=epochs,
    callbacks=[checkpoint])

filename = os.path.join(output, 'report_' + dt + '.html')
plot_hist(history, filename=filename)