from keras.models import model_from_json
from imutils import paths
import numpy as np
import pickle

import cv2
import os

def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            if not file.startswith('warp'):
                normal = os.path.join(path, file)
            else:
                warp = os.path.join(path, file)

    return normal, warp

# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
root_path = os.path.dirname(__file__)

dir_path = os.path.join(path, 'output')
validation_path = [os.path.join(dir_path, folder) for folder in os.listdir(dir_path)][-1]
test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

json_path, warp_json_path = find_file(validation_path, 'json')
weights_path, warp_weights_path = find_file(validation_path, 'h5')

M = np.load('Pickles/M_video1.npy')
M_inv = np.load('Pickles/M_inv_video1.npy')

json = open(json_path, 'r')
model = json.read()
json.close()
model = model_from_json(model)
model.load_weights(weights_path)

# for image in test_list[:10]:
image = test_list[58]
image = cv2.imread(image)
image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
resized = cv2.resize(image, (320, 120))
cv2.imshow('resized', resized)
cv2.waitKey(0)
img = resized[None, ...]
print(img.shape)

prediction = model.predict(img)[0]

left_curve = prediction[:3]
right_curve = prediction[3:]

y = np.linspace(0, resized.shape[0] - 1, resized.shape[0]).astype(int).reshape((-1, 1))
fit_left = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
fit_right = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]
print(prediction)

poly = np.zeros_like(resized)
cv2.imshow('poly', poly)
cv2.waitKey(0)

# left_points = np.array([np.transpose(np.vstack([fit_left, y]))])
# right_points = np.array([np.flipud(np.transpose(np.vstack([fit_right, y])))])

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

for idx, arr in enumerate(points):
    for point in arr:
        cv2.circle(poly, tuple(point), 5, (0, 255, 0), -1)

resized_back=cv2.resize(poly, (1280, 480))
cv2.imshow('resized_back', resized_back)
cv2.waitKey(0)

# newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
#
# result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)