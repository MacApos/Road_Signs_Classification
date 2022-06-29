from lane_detection3.lane_detection import im_show, fit_poly, visualise, generate_points
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


# path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, 'output')
validation_path = os.path.join(dir_path, 'train_2')
# dir_list = os.listdir(dir_path)
# dir_list.sort(key=natural_keys)
# validation_path = [os.path.join(dir_path, folder) for folder in dir_list if folder.startswith('init')][-1]

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
y_range = np.linspace(s_height * 0.6, s_height - 1, 3).astype(int)

# data = pickle.load(open('Pickles/160x80_data.p', 'rb'))
# labels = pickle.load(open('Pickles/160x80_labels.p', 'rb'))
# data = np.array(data)
# labels = np.array(labels)
#
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
#
#
# class train_generator(keras.utils.Sequence):
#     def __init__(self, batch_size, img_size, data_list, labels_list):
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.data_list = data_list
#         self.labels_list = labels_list
#
#     def __len__(self):
#         return len(self.labels_list) // self.batch_size
#
#     def __getitem__(self, idx):
#         i = idx * batch_size
#         data_batch = data[i: i + batch_size]
#         labels_batch = labels[i: i + batch_size]
#         x = np.zeros((batch_size,) + img_size + (3,), dtype='float32')
#         y = np.zeros((batch_size,) + (6, 1), dtype='float32')
#
#         for j, img in enumerate(data_batch):
#             x[j] = img
#
#         for j, label in enumerate(labels_batch):
#             y[j] = label
#
#         return x, y
#
#
# train_datagen = train_generator(batch_size, img_size, x_train, y_train)
#
# # generator check
# for i, (x, y) in enumerate(train_datagen):
#     left_points = np.array(y[i][:3] * s_width).astype(int)
#     right_points = np.array(y[i][3:] * s_width).astype(int)
#
#     for j, y_ in enumerate(y_range):
#         cv2.circle(x[i], (left_points[j][0], y_), 2, (0, 255, 0), -1)
#         cv2.circle(x[i], (right_points[j][0], y_), 2, (0, 255, 0), -1)
#
#     cv2.imshow('img', x[i])
#     cv2.waitKey(0)


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

    coefficients = []
    points_nonzero = []
    lines_nonzero = []

    for arr in points_arr:
        side = np.zeros((s_height, s_width))
        points = np.copy(side)
        coefficients.append(np.polyfit(y_range, arr, 2))
        for j in zip(arr, y_range):
            cv2.circle(points, (j), 4, (255, 0, 0), -1)

        a1, a2 = [j.reshape((-1, 1)) for j in (arr, y_range)]
        con = np.concatenate((a1, a2), axis=1)
        lines = cv2.polylines(np.copy(side), [con], isClosed=False, color=1, thickness=5)

        for nonzero, image in zip([points_nonzero, lines_nonzero], [points, lines]):
            image = cv2.resize(image, (width, height))
            nonzerox = image.nonzero()[1]
            nonzeroy = image.nonzero()[0]
            nonzero.append(nonzerox)
            nonzero.append(nonzeroy)

    output = []

    for nonzero in points_nonzero, lines_nonzero:
        leftx, lefty, rightx, righty = nonzero
        start = min(min(lefty), min(righty))
        left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
        zeros = np.zeros((height, width))
        visualization = visualise(zeros, left_curve, right_curve, start, show_lines=True)
        output.append(visualization)

    # zeros = np.zeros((height, width))
    # print(generate_points(zeros, coefficients[0], coefficients[1], start))

    plt.figure(figsize=(16, 8))
    titles = ['points', 'lines']
    for idx, img in enumerate(output):
        plt.subplot(1, len(output), idx+1)
        plt.title(titles[idx])
        plt.grid(False)
        plt.axis(False)
        plt.imshow(img)
    plt.show()