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

i = 120
img_size = (460, 1280)
frame = cv2.resize(frames[i], img_size[::-1])
label = cv2.resize(labels[i], img_size[::-1])
label = cv2.blur(label, (5, 5))

nonzero = np.nonzero(label)
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

down = min(nonzeroy)
up = max(nonzeroy)
y = np.linspace(min(nonzero[0]), max(nonzero[0]), 5).astype('int')

left = np.zeros((y.shape[0], 2)).astype('int')
right = np.zeros((y.shape[0], 2)).astype('int')

for idx, val in enumerate(y):
    nonzerox = np.nonzero(label[val, :])[0]
    left[idx] = np.array([nonzerox[0], val])
    right[idx] = np.array([nonzerox[-1], val])

empty = []
empty.append(left)
empty.append(right)
points = np.array(empty)

cv2.polylines(frame, points, isClosed=False, color=(0, 0, 255), thickness=4)
cv2.imshow('frame', frame)
cv2.waitKey(0)