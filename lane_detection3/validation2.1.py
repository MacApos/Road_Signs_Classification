from lane_detection3.lane_detection import im_show, fit_poly, visualise
from keras.preprocessing.image import ImageDataGenerator
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
validation_path = os.path.join(dir_path, 'initialized_2')
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
height = width // 2
input_size = cv2.imread(test_list[0]).shape[:-1]
y = np.linspace(s_height * 0.6, s_height - 1, 3).astype(int)



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


def create_points(i):
    # points_arr = np.array(predictions[i] * width).astype(int).reshape((2, -1))
    print(predictions[i] * s_width)

    # nonzero = []
    # for arr in points_arr:
    #     # coefficients = np.polyfit(y, arr, 2)
    #     # for j in zip(arr, y):
    #     #     points = cv2.circle(test[i], (j), 4, 1, -1)
    #
    #     side = np.zeros((s_height, s_width))
    #     a1, a2 = [j.reshape((-1, 1)) for j in (arr, y)]
    #     con = np.concatenate((a1, a2), axis=1)
    #
    #     side = cv2.polylines(side, [con], isClosed=False, color=1, thickness=5)
    #     side = cv2.resize(side, (width, height))
    #     im_show(side)
    #     side = cv2.warpPerspective(side, M_inv, (width, height), flags=cv2.INTER_LINEAR)
    #
    #     nonzerox = side.nonzero()[1]
    #     nonzeroy = side.nonzero()[0]
    #     nonzero.append(nonzerox)
    #     nonzero.append(nonzeroy)
    #
    # leftx, lefty, rightx, righty = nonzero
    #
    # return leftx, lefty, rightx, righty


# def display_lines(i):
    # leftx, lefty, rightx, righty = create_points(i)
    #
    # left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
    #
    # image = cv2.imread(test_list[i])
    # image = cv2.resize(image, (width, height))
    #
    # start = min(min(lefty), min(righty))
    # visualization = visualise(image, left_curve, right_curve, start, show_lines=True)
    # visualization = cv2.resize(visualization, (width, original_image.shape[0]))
    #
    # im_show(visualization)


for i in range(len(test_list[:1])):
    create_points(i)
