import os
import cv2
import random
import pickle
import numpy as np
import PIL
from PIL import ImageOps

import keras
from tensorflow import keras
from keras import layers
from keras import callbacks
from keras.preprocessing.image import load_img, img_to_array, array_to_img

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

batch_size = 32
img_size = (160, 160)
num_classes = 3
epochs = 2

path = r'C:\Nowy folder\10\Praca\Datasets\unet'
data_path = os.path.join(path, 'images')
labels_path = os.path.join(path, r'annotations\trimaps')
output_path = os.path.join(path, 'output')
correct_labels = [label.replace('._', '') for label in os.listdir(labels_path) if label.startswith('._')]

data_list = sorted([os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith('.jpg')])
labels_list = sorted([os.path.join(labels_path, fname) for fname in correct_labels if fname.endswith('.png')])
logs_path = os.path.join(output_path, f'logs.txt')

random.Random(1337).shuffle(data_list)
random.Random(1337).shuffle(labels_list)

def img_csv(name, array):
    array_list = []
    for arr in array:
        if isinstance(arr, np.ndarray):
            arr = arr[0]
        arr = str(arr).replace('.', ',')
        array_list.append(arr)

    df = pd.DataFrame(array_list)
    path = os.path.join('Arrays', name)
    df.to_csv(path, sep='\t', index=False, header=False)

test = labels_list[0]
img = img_to_array(load_img(test))


# # shuffle check
# for data, labels in zip(data_list[:10], labels_list[:10]):
#     print(data, '|', labels)

# random_idx = random.randint(0, len(labels_list))
# img = PIL.ImageOps.autocontrast(load_img(labels_list[random_idx], color_mode='grayscale'))
# img.show()
#
#
class generator(keras.utils.Sequence):
    """All classes have a function called __init__(), which is always executed when the class is being initiated.
    Use the __init__() function to assign values to object properties, or other operations that are necessary to do when
    the object is being created."""

    def __init__(self, batch_size, img_size, data_list, labels_list):
        """The self parameter is a reference to the current instance of the class, and is used to access variables that
        belong to the class."""
        self.batch_size = batch_size
        self.img_size = img_size
        self.data_list = data_list
        self.labels_list = labels_list

    def __len__(self):
        return len(self.labels_list) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_data_paths = self.data_list[i: i + self.batch_size]
        batch_labels_paths = self.labels_list[i: i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='uint8')

        for j, fname in enumerate(batch_data_paths):
            img = load_img(fname, target_size=img_size)
            x[j] = img

        for j, fname in enumerate(batch_labels_paths):
            img = load_img(fname, target_size=img_size, color_mode='grayscale')
            img = np.expand_dims(img, 2)
            # img.shape = (160, 160) → np.expand_dims(img, 2) → img.shape = (160, 160, 1)
            y[j] = img
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1

        return x, y
#
#
# def create_model(img_size, num_classes=1):
#     inputs = layers.Input(shape=img_size + (3,))
#
#     # Entry layers - block 1
#     x = layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#
#     previous_block_activation = x
#
#     # Downsampling - block 2
#     num_filters = 3
#     for filters in [64 * 2 ** i for i in range(num_filters)]:
#         x = layers.Activation('relu')(x)
#         """Separable convolutions consist of first performing a depthwise spatial convolution (which acts on each input
#         channel separately) followed by a pointwise convolution which mixes the resulting output channels."""
#         x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation('relu')(x)
#         x = layers.SeparableConv2D(filters=filters, kernel_size=3, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
#
#         residual = layers.Conv2D(filters=filters, kernel_size=1, strides=2, padding='same')\
#             (previous_block_activation)
#         x = layers.add([x, residual])
#
#         previous_block_activation = x
#
#     # Downsampling - block 3
#     for filters in [256 // 2 ** i for i in range(num_filters+1)]:
#         x = layers.Activation('relu')(x)
#         x = layers.Conv2DTranspose(filters=filters, kernel_size=3, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#
#         x = layers.Activation('relu')(x)
#         x = layers.Conv2DTranspose(filters=filters, kernel_size=3, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#
#         """Upsampling layer for 2D inputs. Repeats the rows and columns of the data by size[0] and size[1]
#         respectively."""
#         x = layers.UpSampling2D(2)(x)
#
#         residual = layers.UpSampling2D(2)(previous_block_activation)
#         residual = layers.Conv2D(filters=filters, kernel_size=1, padding='same', activation='softmax')(residual)
#         x = layers.add([x, residual])
#         previous_block_activation = x
#
#     outputs = layers.Conv2D(filters=num_classes, kernel_size=3, padding='same', activation='softmax')(x)
#     model = keras.Model(inputs, outputs)
#
#     return model
#
#
# keras.backend.clear_session()
#
# model = create_model(img_size, num_classes)
# model.summary()
#
valid_samples = 1000

train_data_list = data_list[:-valid_samples]
train_labels_list = labels_list[:-valid_samples]
valid_data_list = data_list[-valid_samples:]
valid_labels_list = labels_list[-valid_samples:]

pickle.dump(valid_data_list, open(f'Pickles/valid_data_list.p', 'wb'))
pickle.dump(valid_labels_list, open(f'Pickles/valid_labels_list.p', 'wb'))

train_datagen = generator(batch_size, img_size, train_data_list, train_labels_list)
valid_datagen = generator(batch_size, img_size, valid_data_list, valid_labels_list)
print(type(valid_datagen[0][1][0][0][0][0]))

# """sparse_categorical_crossentropy - use this crossentropy loss function when there are two or more label classes.
# We expect labels to be provided as integers."""
# model.compile(optimizer='rmsprop',
#               loss = 'mean_squared_error')
#
# checkpoint_path = os.path.join(output_path, 'checkpoint.h5')
# callback = [callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                       save_best_only=True)]
# csv_logger = callbacks.CSVLogger(logs_path, append=True, separator=';')
# model.fit(train_datagen,
#           epochs=epochs,
#           validation_data=valid_datagen,
#           callbacks=csv_logger)
#
# weights_path = os.path.join(output_path, 'weights.h5')
# model.save(weights_path)