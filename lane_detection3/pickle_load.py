import os
import cv2
import pickle
import random
import numpy as np
from imutils import paths
from datetime import datetime
from lane_detection import im_show, prepare, params, warp_perspective, warp_arr, visualise, visualise_perspective

# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
# labels_path = r'F:\krzysztof\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\labels.p'

path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
labels_path = r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\big_labels.p'

data_path = os.path.join(path, 'data')
data_list = list(paths.list_images(data_path))
labels_list = pickle.load(open(labels_path, 'rb' ))

# file0 = open(path, 'rb')
# lines_dict0 = pickle.load(file0)
# file0.close()

video1 = 2547

print(len(labels_list))

small_labels = []

for label in labels_list:
    coefficient_list = []
    for coefficient in label:
        coefficient = coefficient / 8
        coefficient_list.append(coefficient)

    coefficient_list = np.array(coefficient_list, dtype='float')
    small_labels.append(coefficient_list)

small_labels = np.array(small_labels, dtype='float')

print(small_labels[0], labels_list[0])

random_img = random.sample(data_list[:video1], 1)[0]
random_label = small_labels[data_list.index(random_img)]

left_curve = random_label[:3]
right_curve = random_label[3:]

# print(left_curve, right_curve)

input_shape = (80, 160, 3)
image = cv2.imread(random_img)
width = image.shape[1]
height = image.shape[0]

image = cv2.resize(image, (input_shape[1], input_shape[0]))
frame = np.copy(image)
scalex = input_shape[1] / width
scaley = input_shape[0] / height

# print(scaley, scalex)

number, minpix, margin, video_list = params(width, height)

for video in video_list:
    values = list(video.values())
    name = values[0]
    template = values[1]
    thresh = values[2]
    limit = values[3]

    width = image.shape[1]
    height = image.shape[0]

    src, dst = warp_arr(template, width, height, scalex, scaley)

    img = prepare(image, src, dst, width, height)

    _, M_inv = warp_perspective(frame, dst, src, width, height)

    y = np.linspace(0, height - 1, 15).astype(int).reshape((-1, 1))
    out_img, fit_leftx, fit_rightx, points = visualise(img, y, left_curve, right_curve, False)

    im_show('img', img)

    poly, frame = visualise_perspective(img, points, M_inv, frame)
    im_show('poly', poly)
    # im_show('img', img)
