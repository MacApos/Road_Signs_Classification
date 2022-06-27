import os
import cv2
import random
import pickle
import numpy as np
from imutils import paths
import PIL
from PIL import ImageOps
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras import layers
from tensorflow import keras
from keras import callbacks
from keras.models import load_model
# from keras.utils.image_utils import load_img
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass


path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, 'output')
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

dt = datetime.now().strftime('%d.%m_%H.%M')
# output_path = os.path.join(dir_path, f'initialized_{dt}')
output_path = os.path.join(dir_path, f'initialized_4')
logs_path = os.path.join(output_path, f'logs.txt')
model_path = os.path.join(output_path, 'unet_model.h5')

if not os.path.exists(output_path):
    os.mkdir(output_path)

data = pickle.load(open('Pickles/160x80_data.p', 'rb'))
labels = pickle.load(open('Pickles/160x80_img_labels2.p', 'rb'))


data = np.array(data)
labels = [np.expand_dims(y, 2) for y in labels]
labels = np.array(labels)

batch_size = 32
epochs = 15
img_size = data[0].shape[:-1]
print(img_size)
data, labels = shuffle(data, labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# # load check
# cv2.imshow('img', data[0])
#
# label = np.expand_dims(labels[0], 2)
# label = PIL.ImageOps.autocontrast(array_to_img(label))
# label = img_to_array(label)
# cv2.imshow('label', label)
# cv2.waitKey(1200)
# cv2.destroyAllWindows()

# for img, img_label in zip(x_train[:1], y_train[:1]):
#     poly = np.zeros_like(img).astype('float32')
#     poly[:, :, 1] = img_label[:, :, 0]
#     out_frame = cv2.addWeighted(img, 1, poly, 0.5, 0)
#     cv2.imshow('img', out_frame)
#     cv2.waitKey(0)
#     plt.figure(figsize=(16, 8))
#     for idx, img in enumerate([img, poly, out_frame]):
#         plt.subplot(1, 3, idx+1)
#         plt.grid(False)
#         plt.axis(False)
#         imgplot = plt.imshow(img[:,:,::-1])
#     plt.show()


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
        y = np.zeros((batch_size,) + img_size + (1,), dtype='uint8')

        for j, img in enumerate(data_batch):
            x[j] = img

        for j, img in enumerate(labels_batch):
            # img.shape = (160, 160) → np.expand_dims(img, 2) → img.shape = (160, 160, 1)
            img = np.expand_dims(img, 2)
            y[j] = img

        return x, y


def create_model(img_size, num_classes):
    inputs = layers.Input(shape=img_size + (3,))

    # Entry layers - block 1
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x

    # Downsampling - block 2
    num_filters = 3
    start = 64
    block2 = [start * 2 ** i for i in range(num_filters)]
    for filters in block2:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        residual = layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding="same")\
            (previous_block_activation)
        x = layers.add([x, residual])

        previous_block_activation = x

    # Downsampling - block 3
    block3 = [block2[-1] // 2 ** i for i in range(num_filters + 1)]
    print(block2, block3)
    for filters in block3:
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

    outputs = layers.Conv2D(filters=num_classes, kernel_size=3, padding='same', activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    return model


keras.backend.clear_session()

model = create_model(img_size, 2)
model.summary()


print(x_train.shape, y_train.shape)

train_generator = ImageDataGenerator()
valid_generator = ImageDataGenerator()

train_datagen = train_generator.flow(x=x_train, y=y_train, batch_size=batch_size)
valid_datagen = valid_generator.flow(x=x_test, y=y_test, batch_size=batch_size)

# train_datagen = generator(batch_size, img_size, x_train, y_train)
# valid_datagen = generator(batch_size, img_size, x_test, y_test)

# # generator check
# for x, y in train_datagen:
#     image = x[0]
#
#     label = PIL.ImageOps.autocontrast(array_to_img(y[0]))
#     label = img_to_array(label)
#
#     cv2.imshow('x', image)
#     cv2.imshow('y', label)
#     cv2.waitKey(1200)
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

loss = 'sparse_categorical_crossentropy'
model.compile(optimizer = 'rmsprop',
              loss = loss)

csv_logger = callbacks.CSVLogger(logs_path, append=True, separator='\t')
model.fit(train_datagen,
          epochs=epochs,
          validation_data=valid_datagen,
          callbacks=csv_logger
          )

logs = open(logs_path, 'a')
logs.write(f'\nepochs = {epochs}\n')
logs.write(f'batch_size = {batch_size}\n')
logs.write(f'input_shape = {img_size}\n')
logs.write(f'loss = {loss}\n')
logs.close()

model.save(model_path)
predictions = model.predict(valid_datagen)

def display_mask(i):
    mask = np.argmax(predictions[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    image = PIL.ImageOps.autocontrast(array_to_img(mask))
    # image.show()
    img = img_to_array(image)
    cv2.imshow('predictions', img)
    cv2.waitKey(500)

for i in range(2):
    display_mask(i)