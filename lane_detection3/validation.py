from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from imutils import paths
import numpy as np
import pickle
import cv2
import re
import os

def get_digits0(s):
    head = s.rstrip('0123456789')
    return s[len(head):]


def get_digits1(text):
    alpha = text.strip('0123456789')
    print(alpha)
    return text.split(alpha)


def get_digits2(text):
    digits = ''
    for t in text:
        if not digits:
            if t.isdigit():
                digits += t
        elif t.isdigit() and len(digits):
            digits += t
    return digits

def natural_keys(text):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]

def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)

path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
M = np.load('Pickles/M_video1.npy')
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
dir_path = os.path.join(path, 'output')
dir_list = os.listdir(dir_path)
dir_list.sort(key=natural_keys)
validation_path = [os.path.join(dir_path, folder) for folder in dir_list if folder.startswith('init')][-1]

test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

model_path = find_file(validation_path, 'h5')
model = keras.models.load_model(model_path)

batch_size = 32
height = 80
width = 160
input_shape = (height, width, 3)

save_path = f'Pickles/{width}x{height}_test.p'

if not os.path.exists(save_path):
    test = []
    for path in test_list:
        image = cv2.imread(path)
        image = cv2.resize(image, (image.shape[1], image.shape[1]//2))
        warp = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        img = cv2.resize(warp, (width, height)) / 255
        test.append(img)
    pickle.dump(test, open(save_path, 'wb'))

else:
    test = pickle.load(open('Pickles/160x80_warp_data.p', 'rb'))

test = np.array(test)

test_generator = ImageDataGenerator()
test_datagen = test_generator.flow(x=test, batch_size=batch_size, shuffle=False)

# generator check
# for x in test_datagen:
#     for i in x:
#         cv2.imshow('test', test[0])
#         cv2.imshow('test1', i)
#         cv2.waitKey(0)

for i in range(len(test_list)):
    out_img = test[i]
    predictions = model.predict(out_img[None,:,:,:])[0]
    print(predictions)

    y = np.linspace(0, height-1, 3).astype(int)

    # for i in range(len(test_list)):
    left_curve = np.array(predictions[:3] * width).astype(int)
    right_curve = np.array(predictions[3:] * width).astype(int)

    print(left_curve, right_curve)

    for k, y_ in enumerate(y):
        cv2.circle(out_img, (left_curve[k], y_), 4, (0, 255, 0), -1)
        cv2.circle(out_img, (right_curve[k], y_), 4, (0, 255, 0), -1)

    cv2.imshow('out_img', out_img)
    cv2.waitKey(0)

#
# y = np.linspace(0, resized.shape[0] - 1, resized.shape[0]).astype(int).reshape((-1, 1))
# fit_left = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
# fit_right = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]
# print(prediction)
#
# poly = np.zeros_like(resized)
# cv2.imshow('poly', poly)
# cv2.waitKey(0)
#
# # left_points = np.array([np.transpose(np.vstack([fit_left, y]))])
# # right_points = np.array([np.flipud(np.transpose(np.vstack([fit_right, y])))])
#
# empty = []
# flipud = False
#
# for arr in fit_left, fit_right:
#     arr = arr.astype(int)
#     con = np.concatenate((arr, y), axis=1)
#
#     if flipud:
#         con = np.flipud(con)
#
#     flipud = True
#     empty.append(con)
#
# points = np.array(empty)
#
# for idx, arr in enumerate(points):
#     for point in arr:
#         cv2.circle(poly, tuple(point), 5, (0, 255, 0), -1)
#
# resized_back=cv2.resize(poly, (1280, 480))
# cv2.imshow('resized_back', resized_back)
# cv2.waitKey(0)

# newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
#
# result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)