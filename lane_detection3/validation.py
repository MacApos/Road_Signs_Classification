from lane_detection3.lane_detection import im_show, fit_poly, visualise
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from imutils import paths
import numpy as np
import pickle
import cv2
import re
import os


def get_digits0(s):
    head = s.rstrip('0123456789')
    return s[len(head):]


def get_digits1(text):
    alpha = text.strip('0123456789')
    print(alpha)
    return text.split(alpha)


def get_digits2(text):
    digits = ''
    for t in text:
        if not digits:
            if t.isdigit():
                digits += t
        elif t.isdigit() and len(digits):
            digits += t
    return digits


def natural_keys(text):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]


def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)


path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, 'output')
dir_list = os.listdir(dir_path)
dir_list.sort(key=natural_keys)
validation_path = [os.path.join(dir_path, folder) for folder in dir_list if folder.startswith('init')][-1]

test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

model_path = find_file(validation_path, 'h5')
model = keras.models.load_model(model_path)

M = np.load('Pickles/M_video1.npy')
M_inv = np.load('Pickles/M_inv_video1.npy')

batch_size = 32
s_width = 160
s_height = 80
img_size = (s_height, s_width)
original_image = cv2.imread(test_list[0])
width = original_image.shape[1]
height = width // 2
y = np.linspace(0, s_height - 1, 3).astype(int)

save_path = f'Pickles/{s_width}x{s_height}_test.p'

if not os.path.exists(save_path):
    test = []
    for path in test_list:
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))
        warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
        img = cv2.resize(warp, (s_width, s_height)) / 255
        test.append(img)
        pickle.dump(test, open(save_path, 'wb'))
else:
    test = pickle.load(open('Pickles/160x80_warp_data.p', 'rb'))

test = np.array(test)
test_generator = ImageDataGenerator()
test_datagen = test_generator.flow(x=test, batch_size=batch_size, shuffle=False)

# # generator check
# for x in test_datagen:
#     for i in x:
#         cv2.imshow('test', test[0])
#         cv2.imshow('test1', i)
#         cv2.waitKey(0)

predictions = model.predict(test_datagen)


def create_points(i):
    points_arr = np.array(predictions[i] * s_width).astype(int).reshape((2, -1))

    nonzero = []
    for arr in points_arr:
        # coefficients = np.polyfit(y, arr, 2)
        # for j in zip(arr, y):
        #     points = cv2.circle(test[i], (j), 4, 1, -1)

        side = np.zeros((s_height, s_width))
        a1, a2 = [j.reshape((-1, 1)) for j in (arr, y)]
        con = np.concatenate((a1, a2), axis=1)

        side = cv2.polylines(side, [con], isClosed=False, color=1, thickness=5)
        side = cv2.resize(side, (width, height))
        side = cv2.warpPerspective(side, M_inv, (width, height), flags=cv2.INTER_LINEAR)

        nonzerox = side.nonzero()[1]
        nonzeroy = side.nonzero()[0]
        nonzero.append(nonzerox)
        nonzero.append(nonzeroy)

    leftx, lefty, rightx, righty = nonzero

    return leftx, lefty, rightx, righty


def display_lines(i):
    leftx, lefty, rightx, righty = create_points(i)

    left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)

    image = cv2.imread(test_list[i])
    image = cv2.resize(image, (width, height))

    start = min(min(lefty), min(righty))
    visualization = visualise(image, left_curve, right_curve, start, show_lines=True)
    visualization = cv2.resize(visualization, (width, original_image.shape[0]))

    im_show(visualization)


for i in range(test.shape[0]):
    display_lines(i)
