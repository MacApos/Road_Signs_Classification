import numpy as np
import os
import cv2
import glob
import pickle
import keras
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

""" This file contains the convolutional neural network for detecting
lane lines on a perspective transformed image. This file currently contains
the necessary code to 1) load the perspective transformed images and labels
pickle files, 2) shuffles and then splits the data into training and validation
sets, 3) creates the neural network architecture, 4) trains the network,
5) saves down the model architecture and weights, 6) shows the model summary.
"""
dt = datetime.now().strftime('%d.%m_%H.%M.%S')
path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
dir_path = os.path.join(path, 'output')
output_path = os.path.join(dir_path, f'initialized_{dt}')

data = pickle.load(open('Pickles/160x80_warp_data.p', 'rb'))
labels = pickle.load(open('Pickles/160x80_warp_labels.p', 'rb'))

data = np.array(data)
labels = np.array(labels)

train_images, labels = shuffle(data, labels)
x_train, x_test, y_train, y_test = train_test_split(train_images, labels, test_size=0.1)

# load check
from lane_detection import im_show, visualise

for idx, image in enumerate(x_train[:1]):
    left_curve = y_train[idx][:3]
    right_curve = y_train[idx][3:]
    print(left_curve, right_curve)
    warp = visualise(image, left_curve, right_curve, image.shape[0]*0, show_lines=True)
    im_show(warp)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
batch_size = 150
epochs = 10
pool_size = (2, 2)
input_shape = x_train.shape[1:]

# Here is the actual neural network
model = Sequential()
# Normalizes incoming inputs. First layer needs the input shape to work
model.add(BatchNormalization(input_shape=input_shape))

# Conv Layer 1
model.add(Convolution2D(64, (3, 3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Conv Layer 2
model.add(Convolution2D(32, (3, 3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Conv Layer 3
model.add(Convolution2D(16, (3, 3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Conv Layer 4
model.add(Convolution2D(8, (3, 3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=pool_size))

# Flatten and Dropout
model.add(Flatten())
model.add(Dropout(0.5))

# FC Layer 1
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# FC Layer 2
model.add(Dense(64))
model.add(Activation('relu'))

# FC Layer 3
model.add(Dense(32))
model.add(Activation('relu'))

# Final FC Layer - six outputs - the three coefficients for each of the two lane lines polynomials
model.add(Dense(6))

# Using a generator to help the model generalize/train better
datagen = ImageDataGenerator(rotation_range = 10, vertical_flip = True, height_shift_range = .1)
datagen.fit(x_train)

# Compiling and training the model
# Currently using MAE instead of MSE as MSE tends to only have 1 label for left curve, 1 for right curve, and 1 for straight (nothing in between)
model.compile(optimizer='Adam', loss='mean_absolute_error')
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch = len(x_train),
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

# Show summary of model
model.summary()
