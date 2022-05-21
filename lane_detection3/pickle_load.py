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

path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
labels_path = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\big_labels.p'

data_path = os.path.join(path, 'data')
data_list = list(paths.list_images(data_path))
labels_list = pickle.load(open(labels_path, 'rb' ))

# file0 = open(path, 'rb')
# lines_dict0 = pickle.load(file0)
# file0.close()

small_labels = []
scale_factor = 1/4

for label in labels_list:
    coefficient_list = []
    for coefficient in label:
        print(coefficient)
        coefficient = coefficient * scale_factor
        coefficient_list.append(coefficient)

    coefficient_list = np.array(coefficient_list, dtype='float')
    small_labels.append(coefficient_list)

small_labels = np.array(small_labels, dtype='float')

# random_img = random.sample(data_list, 1)[0]
# random_img = data_list[0]
# random_label = small_labels[data_list.index(random_img)]

for idx, data in enumerate(data_list):
    left_curve = small_labels[idx][:3]
    right_curve = small_labels[idx][3:]

    image = cv2.imread(data)
    image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))
    frame = np.copy(image)
    width = image.shape[1]
    height = image.shape[0]

    number, minpix, margin, video_list = params(width, height)

    for video in video_list:
        values = list(video.values())
        template = values[1]

        src, dst = warp_arr(template, width, height)

        img = prepare(image, src, dst, width, height)

        _, M_inv = warp_perspective(frame, dst, src, width, height)

        out_img, fit_leftx, fit_rightx, points = visualise(height, np.dstack((img, img, img)), left_curve, right_curve)

        poly, frame = visualise_perspective(img, points, M_inv, frame)
        im_show(out_img)
