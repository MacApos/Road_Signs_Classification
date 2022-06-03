import os
import cv2
import pickle
import random
import numpy as np
from imutils import paths
from datetime import datetime
from lane_detection import im_show, params, visualise_perspective

path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
labels_path = r'F:\krzysztof\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\labels.p'

# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# labels_path = r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\160x60_labels.p'

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
        coefficient = coefficient * scale_factor
        coefficient_list.append(coefficient)

    coefficient_list = np.array(coefficient_list, dtype='float')
    small_labels.append(coefficient_list)

small_labels = np.array(small_labels, dtype='float')

# random_img = random.sample(data_list, 1)[0]
# random_img = data_list[0]
# random_label = small_labels[data_list.index(random_img)]

image = cv2.imread(data_list[0])
image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))
width = image.shape[1]
height = image.shape[0]

video_list, dst = params(width, height)

i=0
for video in video_list:
    name = video['name']
    thresh = video['thresh']
    limit = video['limit']
    src = video['src']

    for idx, path in enumerate(data_list[: i + limit]):
        left_curve = labels_list[idx][:3]
        right_curve = labels_list[idx][3:]
        image = cv2.imread(path)
        image = cv2.resize(image, (width, height))

        poly, frame = visualise_perspective(image, left_curve, right_curve, src, dst)
        im_show(frame)

    i += limit
