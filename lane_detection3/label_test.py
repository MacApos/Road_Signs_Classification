import numpy as np
import random
import pickle
import cv2
import os

data = pickle.load(open('Pickles/160x80_warp_data.p', 'rb'))
labels = pickle.load(open('Pickles/160x80_warp_labels.p', 'rb'))
coefficients = pickle.load(open('Pickles/160x80_warp_coefficients.p', 'rb'))

height = data[0].shape[0]

# load check
from lane_detection3.lane_detection import im_show, visualise

y = np.linspace((height - 1)*0.6, height, 3).astype(int)
for i, image in enumerate(data[5:]):
    left_points = labels[i+5][:3]
    right_points = labels[i+5][3:]

    left_curve = coefficients[i+5][:3]
    right_curve = coefficients[i+5][3:]

    warp = visualise(image, left_curve, right_curve, (height - 1)*0.6, show_lines=True)
    im_show(warp)
    for j, y_ in enumerate(y):
        cv2.circle(warp, (left_points[j], y_), 2, (0, 255, 0), -1)
        cv2.circle(warp, (right_points[j], y_), 2, (0, 255, 0), -1)
        #
    im_show(warp)
