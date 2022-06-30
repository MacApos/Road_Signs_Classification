from lane_detection3.lane_detection import im_show, fit_poly, visualise
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from imutils import paths
import numpy as np
import pickle
import cv2
import re
import os


def natural_keys(text):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]


def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)


path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, 'output')
validation_path = os.path.join(dir_path, 'train_2')

test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

model_path = find_file(validation_path, 'h5')
model = keras.models.load_model(model_path)

batch_size = 100
s_width = 160
s_height = 80
img_size = (s_height, s_width)
original_image = cv2.imread(test_list[0])
width = original_image.shape[1]
height = original_image.shape[0]
print(width, height)
input_size = cv2.imread(test_list[0]).shape[:-1]

y_range = np.linspace(s_height * 0.65, s_height * 0.85, 3).astype(int)


class generator(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, test_list):
        self.batch_size = batch_size
        self.img_size = img_size
        self.test_list = test_list

    def __len__(self):
        return len(self.test_list) // self.batch_size

    def __getitem__(self, idx):
        i = idx * batch_size
        test_batch = test_list[i: i + batch_size]
        x = np.zeros((batch_size,) + img_size + (3,), dtype='float32')

        for j, path in enumerate(test_batch):
            img = cv2.imread(path)
            img = cv2.resize(img, img_size[::-1]) / 255
            x[j] = img

        return x


train_datagen = generator(batch_size, img_size, test_list)
predictions = model.predict(train_datagen)

for i in range(len(test_list)):
    points_arr = np.array(predictions[i] * s_width).astype(int).reshape((2, -1))

    mask = np.zeros((height, width))
    nonzero = []
    for arr in points_arr:
        side = np.zeros((s_height, s_width))
        for j in zip(arr, y_range):
            cv2.circle(np.copy(side), (j), 4, (255, 0, 0), -1)

        a1, a2 = [j.reshape((-1, 1)) for j in (arr, y_range)]
        con = np.concatenate((a1, a2), axis=1)
        lines = cv2.polylines(np.copy(side), [con], isClosed=False, color=1, thickness=5)
        side = cv2.resize(lines, (width, height))
        mask += side

        nonzerox = side.nonzero()[1]
        nonzeroy = side.nonzero()[0]
        nonzero.append(nonzerox)
        nonzero.append(nonzeroy)

    leftx, lefty, rightx, righty = nonzero
    left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)

    start = min(min(lefty), min(righty))
    stop = max(max(lefty), max(righty))

    test_image = cv2.imread(test_list[i])
    test_image = test_image / 255
    zeros = np.zeros_like(mask)
    poly = np.dstack((zeros, mask, zeros))

    print(poly.shape, test_image.shape)

    # prediction = cv2.addWeighted(test_image, 1, poly, 0.5, 0)
    out_img = visualise(test_image, left_curve, right_curve, 0.6*height, 1*height)
    cv2.imshow('prediction', out_img)
    cv2.waitKey(0)