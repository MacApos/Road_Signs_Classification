import os
import cv2
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

classes = 43
data_path = os.getcwd()

file1 = os.path.join(data_path, '../data.npy')
file2 = os.path.join(data_path, '../labels.npy')

if not os.path.exists(file1) or not os.path.exists(file2):
    data = []
    labels = []
    for i in range(classes):
        path = os.path.join(data_path, '../archive/Train', str(i))
        images = os.listdir(path)

        print('doing')
        for j in images:
            image = Image.open(path+'\\'+j)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)

    data = np.array(data)
    labels = np.array(labels)
    np.save(file1, data, allow_pickle=True, fix_imports=True)
    np.save(file2, labels, allow_pickle=True, fix_imports=True)

else:
    print('not doing')

data = np.load(file1)
labels = np.load(file2)

# print(data.shape, labels.shape)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=15, validation_data=(X_test, y_test))

dt = datetime.now().strftime('%d_%m_%Y_%H_%M')
model.save('model'+dt+'.hdf5')

plt.figure(0)
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
