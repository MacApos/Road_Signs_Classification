import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# data_npy = r'F:\krzysztof\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\data.npy'
# data = np.load(data_npy)

# print(data.shape)
# image = data[1]
#
# # image = cv2.imread(image)
# cv2.imshow('image', image)
# cv2.waitKey(0)
def make_input(message):
    print(message, ' [y/n]')
    x = input()
    x = x.lower()
    if x != 'y' and x != 'n':
        raise Exception('Invalid input')

    return x

from PIL import Image
import tensorflow as tf
img_cv = cv2.imread(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pictures\perspective.jpg')
img_plt = mpimg.imread(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pictures\perspective.jpg')
print(type(img_plt))

#
# img = tf.keras.preprocessing.image.array_to_img(img_data)
# plt.imshow(img)
# plt.show()
#
# array = tf.keras.preprocessing.image.img_to_array(img)
# # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
# plt.imshow(array)
# plt.show()
#
# cv2.waitKey(0)