import os
import cv2
import random
import pickle
import numpy as np
import PIL
from PIL import ImageOps
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras
from tensorflow import keras
from keras import layers
from keras import callbacks
from keras.preprocessing.image import load_img, ImageDataGenerator

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

batch_size = 32
num_classes = 3
epochs = 2

path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, 'output')
data_path = os.path.join(path, 'train')

if not os.path.exists(dir_path):
    os.mkdir(dir_path)

dt = datetime.now().strftime('%d.%m_%H.%M')
output_path = os.path.join(dir_path, f'initialized_{dt}')
logs_path = os.path.join(output_path, f'logs.txt')

if not os.path.exists(output_path):
    os.mkdir(output_path)

data = np.load('Pickles/160x60_data.npy')
labels = np.load('Pickles/160x60_img_labels.npy')

cv2.imshow('img', labels[0])
cv2.waitKey(0)

img_size = data.shape[1:-1]
data, labels = shuffle(data, labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# load check
# for img, label in zip(x_train[:1], y_train[:1]):
#     poly = np.dstack((label, label, label))
#     poly[:, :, [0, 2]] = 0
#     out_frame = cv2.addWeighted(img, 1, poly, 0.5, 0)
#     plt.figure(figsize=(16, 8))
#     for idx, img in enumerate([img, poly, out_frame]):
#         plt.subplot(1, 3, idx+1)
#         plt.grid(False)
#         plt.axis(False)
#         imgplot = plt.imshow(img[:,:,::-1])
#     plt.show()

# random_idx = random.randint(0, len(labels_list))
# img = PIL.ImageOps.autocontrast(load_img(labels_list[random_idx], color_mode='grayscale'))
# img.show()

class generator(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, data_list, labels_list):
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_list = data_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.labels_list) // self.batch_size

    def __getitem__(self, idx):
        i = idx * batch_size
        data_batch = data[i: i + batch_size]
        labels_batch = labels[i: i + batch_size]
        x = np.zeros((batch_size,) + img_size + (3,), dtype='float32')
        y = np.zeros((batch_size,) + img_size + (1,), dtype='float32')

        for j, img in enumerate(data_batch):
            x[j] = img

        for j, img in enumerate(labels_batch):
            # img.shape = (160, 160) → np.expand_dims(img, 2) → img.shape = (160, 160, 1)
            img = np.expand_dims(img, 2)
            y[j] = img

        return x, y


def create_model(img_size, num_classes=1):
    inputs = layers.Input(shape=img_size + (3,))

    # Entry layers - block 1
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x

    # Downsampling - block 2
    num_filters = 3
    for filters in [64 * 2 ** i for i in range(num_filters)]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        residual = layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding="same", activation='softmax')\
            (previous_block_activation)
        x = layers.add([x, residual])

        previous_block_activation = x

    # Downsampling - block 3
    for filters in [256 // 2 ** i for i in range(num_filters+1)]:
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters=filters, kernel_size=1, padding='same')(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(filters=num_classes, kernel_size=3, padding='same')(x)
    model = keras.Model(inputs, outputs)

    return model

keras.backend.clear_session()

model = create_model(img_size, num_classes)
model.summary()

# train_datagen = generator(batch_size, img_size, x_train, y_train)
# valid_datagen = generator(batch_size, img_size, x_test, y_test)

train_datagen = ImageDataGenerator(channel_shift_range=0.2)
valid_datagen = ImageDataGenerator(rescale=1. / 255.)

train_datagen = train_datagen.flow(x=x_train, y=y_train, batch_size=batch_size)
valid_datagen = valid_datagen.flow(x=x_test, y=y_test, batch_size=batch_size)

# # generator check
# for x, y in train_datagen:
#     image = x[0]
#     label = y[0]
#     cv2.imshow('x', label)
#     cv2.waitKey(0)
#
#     poly = np.dstack((label, label, label))
#     poly[:, :, [0, 2]] = 0
#     out_frame = cv2.addWeighted(image, 1, poly, 0.5, 0)
#     plt.figure(figsize=(16, 8))
#     for idx, img in enumerate([image, poly, out_frame]):
#         plt.subplot(1, 3, idx + 1)
#         plt.grid(False)
#         plt.axis(False)
#         imgplot = plt.imshow(img[:, :, ::-1])
#     break
# plt.show()

model.compile(optimizer='rmsprop',
              loss = 'sparse_categorical_crossentropy')

checkpoint_path = os.path.join(output_path, 'checkpoint.h5')
callback = [callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                      save_best_only=True)]
csv_logger = callbacks.CSVLogger(logs_path, append=True, separator=';')

model.fit(train_datagen,
          epochs=epochs,
          validation_data=valid_datagen,
          callbacks=csv_logger)

weights_path = os.path.join(output_path, 'weights.h5')
model.save(weights_path)