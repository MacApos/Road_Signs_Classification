import os
import cv2
import pickle
import shutil
import numpy as np
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt


def to_csv(name, array):
    array_list = []
    for arr in array:
        if isinstance(arr, np.ndarray):
            arr = arr[0]
        arr = str(arr).replace('.', ',')
        array_list.append(arr)

    df = pd.DataFrame(array_list)
    # path = os.path.join(r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3\Arrays', name)
    path = os.path.join(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Arrays', name)
    df.to_csv(path, sep='\t', index=False, header=False)


path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'

frames_list = pickle.load(open('Pickles/160x80_data.p', 'rb'))
labels_list = pickle.load(open('Pickles/160x80_img_labels.p', 'rb'))

cv2.imshow('label', frames_list[0])
cv2.waitKey(0)
print(labels_list[0].shape)

img_size = (1280, 460)
frame = cv2.resize(frames_list[0], img_size)
label = cv2.resize(labels_list[0], img_size)

to_csv('label', label)

# cv2.imshow('frame', frame)
nonzerox = np.nonzero(frame)[1]
nonzeroy = np.nonzero(frame)[0]
print(min(nonzerox),max(nonzerox))

print(np.transpose(np.nonzero(label)))

