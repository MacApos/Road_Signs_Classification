import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import pickle
import cv2
import os


path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'

data_path = os.path.join(path, 'train')
data_list = list(paths.list_images(data_path))


mtx = pickle.load(open('Pickles/mtx.p', 'rb'))
dist = pickle.load(open('Pickles/dist.p', 'rb'))

for i in range(len(data_list[:60])):
    image = cv2.imread(data_list[i])
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)

    cv2.imwrite(f'Pictures/camera_calibration/road_{i}_0.jpg', image)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    cv2.imwrite(f'Pictures/camera_calibration/road_{i}_1.jpg', undistorted)
    # cv2.imshow(f'undistorted', undistorted)
    # cv2.waitKey(0)
    #
    # plt.subplot(1,2,1)
    # plt.imshow(image)
    #
    # plt.subplot(1,2,2)
    # plt.imshow(undistorted)
    # plt.show()