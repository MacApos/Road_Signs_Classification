import os
import cv2
import pickle
import shutil
import numpy as np
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt


def to_csv(name, arr):
    df = pd.DataFrame(arr)
    arrays_path = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3\Arrays'
    path = os.path.join(arrays_path, name)
    df.to_csv(path, sep='\t', index=False, header=False)

frames = pickle.load(open('Pickles/160x80_data.p', 'rb'))
labels = pickle.load(open('Pickles/160x80_unet_labels.p', 'rb'))

i = 150
img_size = (640, 1280)
frame = cv2.resize(frames[i], img_size[::-1])
label = cv2.resize(labels[i], img_size[::-1])
label = cv2.blur(label, (5, 5))
cv2.imshow('label', label)
cv2.waitKey(0)

def draw_lines(mask, image):
    nonzero = np.nonzero(mask)
    y = np.linspace(min(nonzero[0]), max(nonzero[0]), 15).astype(int)

    leftx = np.zeros_like(y).astype(int)
    rightx = np.zeros_like(y).astype(int)
#
    for idx, val in enumerate(y):
        nonzerox = np.nonzero(mask[val, :])[0]
        leftx[idx] = np.array([nonzerox[0]])
        rightx[idx] = np.array([nonzerox[-1]])

    leftx_start = leftx[np.argmax(y)]
    rightx_start = rightx[np.argmax(y)]

    left_curve = np.polyfit(y, leftx, 2)
    right_curve = np.polyfit(y, rightx, 2)

    y = y.reshape((-1,1))
    fit_left = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
    fit_right = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]

    center = image.shape[1] // 2
    cv2.circle(image, (leftx_start, np.max(y)), 10, (255, 0, 0), -1)
    cv2.circle(image, (rightx_start, np.max(y)), 10, (255, 0, 0), -1)
    cv2.circle(image, (center, np.max(y)), 10, (0, 255, 0), -1)

    cv2.imshow('image', image)
    cv2.waitKey(0)

    print(type(leftx_start), leftx_start)
    offset = (rightx_start - center) - (center - leftx_start)
    print(offset)

    empty = []
    flipud = False

    for arr in fit_left, fit_right:
        arr = arr.astype(int)
        con = np.concatenate((arr, y), axis=1)

        if flipud:
            con = np.flipud(con)

        flipud = True
        empty.append(con)

    points = np.array(empty)

    cv2.polylines(image, points, isClosed=False, color=(0, 0, 255), thickness=4)
    cv2.imshow('image', image)
    cv2.waitKey(0)

lines = draw_lines(label, frame)

