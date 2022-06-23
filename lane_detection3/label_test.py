import numpy as np
import random
import pickle
import cv2

data = pickle.load(open('Pickles/160x80_warp_data.p', 'rb'))
labels = pickle.load(open('Pickles/160x80_warp_labels.p', 'rb'))

data = np.array(data)
labels = np.array(labels)

random_idx = random.randint(0, data.shape[0])
image = data[random_idx]
left_label = labels[random_idx][:3]
right_label = labels[random_idx][3:]

height = image.shape[0]
width = image.shape[1]

# load check
from lane_detection3.lane_detection import im_show, visualise

y = np.linspace(0, height - 1, 3).astype(int)
for i, image in enumerate(data[random_idx:random_idx+10]):
    left = labels[random_idx+i][:3]
    right = labels[random_idx+i][3:]
    fit_left = np.array(left[0] * y ** 2 +
                        left[1] * y +
                        left[2]).astype(int)
    fit_right = np.array(right[0] * y ** 2 +
                         right[1] * y +
                         right[2]).astype(int)
    warp = visualise(image, left, right, show_lines=True)

    print(fit_left, fit_right)
    for j, y_ in enumerate(y):
        cv2.circle(warp, (fit_left[j], y_), 2, (0, 255, 0), -1)
        cv2.circle(warp, (fit_right[j], y_), 2, (0, 255, 0), -1)

    im_show(warp)


print(y)
zeros = np.zeros((3,3))

for i in range(1, 3):
    zeros[i] = [j**i for j in np.linspace(0, height, 3)]

zeros = np.flipud(zeros)




print(fit_left)