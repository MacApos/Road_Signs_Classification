from imutils import paths
import numpy as np
import pickle
import cv2
import os


def sort_path(path):
    sorted_path = []
    for file in os.listdir(path):
        number = int(''.join(n for n in file if n.isdigit()))
        sorted_path.append(number)

    sorted_path = sorted(sorted_path)
    return [path + fr'\{str(f)}.jpg' for f in sorted_path]


road_images_path = r'F:\Nowy folder\10\Praca\Datasets\Video_data\frames'
labels = pickle.load(open( "Pickles/lane_labels.p", "rb" ))

road_images = sort_path(road_images_path)

image = cv2.imread(road_images[0])
height = image.shape[0]
width = image.shape[1]


for label in labels[:2]:
    left_coeff = label[:3]
    right_coeff = label[3:]

    y = np.linspace(0, height-1, height)
    fit_leftx = left_coeff[0] * y ** 2 + left_coeff[1] * y + left_coeff[2]
    fit_rightx = right_coeff[0] * y ** 2 + right_coeff[1] * y + right_coeff[2]