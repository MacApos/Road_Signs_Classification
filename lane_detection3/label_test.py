import numpy as np
import random
import pickle
import cv2
import os

data = pickle.load(open('Pickles/160x80_warp_data.p', 'rb'))
labels = pickle.load(open('Pickles/160x80_warp_labels.p', 'rb'))
# coefficients = pickle.load(open('Pickles/160x80_coefficients.p'))
#
#
# data = np.array(data)
# labels = np.array(labels)
#
# random_idx = random.randint(0, data.shape[0])
# image = data[random_idx]
# left_label = labels[random_idx][:3]
# right_label = labels[random_idx][3:]
#
# print(left_label, right_label)
#
# height = image.shape[0]
# width = image.shape[1]
#
# # load check
# from lane_detection3.lane_detection import im_show, visualise
#
# y = np.linspace(0, height - 1, 3).astype(int)
# for i, image in enumerate(data[random_idx:random_idx+10]):
#     left_points = labels[random_idx+i][:3]
#     right_points = labels[random_idx+i][3:]
#
#     left_curve = coefficients[random_idx+i][:3]
#     right_curve = coefficients[random_idx + i][3:]
#
#     im_show(image)
#     warp = visualise(image, left_curve, right_curve, show_lines=True)
#     for j, y_ in enumerate(y):
#         cv2.circle(warp, (left_points[j], y_), 2, (0, 255, 0), -1)
#         cv2.circle(warp, (right_points[j], y_), 2, (0, 255, 0), -1)
#
#     im_show(warp)
# #
# #
# print(y)
# zeros = np.zeros((3,3))
#
# for i in range(1, 3):
#     zeros[i] = [j**i for j in np.linspace(0, height, 3)]
#
# zeros = np.flipud(zeros)
#
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
#
# frames_path = os.path.join(path, 'frames')
# labels_path = os.path.join(path, 'labels')
#
# check = r'C:\Nowy folder\10\Praca\Datasets\Video_data\labels\00510.jpg'
#
# print(os.path.exists(check))