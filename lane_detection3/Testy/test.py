import numpy as np
import cv2

data_npy = r'F:\krzysztof\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\data.npy'
data = np.load(data_npy)

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

make_input('Procced?')