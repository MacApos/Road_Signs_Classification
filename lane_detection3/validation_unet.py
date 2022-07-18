import os
import cv2
import pickle
import shutil
import numpy as np
from imutils import paths
import PIL
from PIL import ImageOps
from tensorflow import keras
from lane_detection3.lane_detection import visualise
# from keras.utils import img_to_array, array_to_img
from keras.preprocessing.image import img_to_array, array_to_img


def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)


path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, 'output')
test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

batch_size = 32
img_size = (80, 160)
input_size = cv2.imread(test_list[0]).shape[:-1]


class generator(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, test_list):
        self.batch_size = batch_size
        self.img_size = img_size
        self.test_list = test_list

    def __len__(self):
        return len(self.test_list) // self.batch_size

    def __getitem__(self, idx):
        i = idx * batch_size
        test_batch = test_list[i: i + batch_size]
        x = np.zeros((batch_size,) + img_size + (3,), dtype='float32')

        for j, path in enumerate(test_batch):
            img = cv2.imread(path)
            img = cv2.resize(img, img_size[::-1]) / 255
            x[j] = img

        return x


def choose_labels(fname):
    validation_path = os.path.join(dir_path, fname)
    model_path = find_file(validation_path, 'h5')
    model = keras.models.load_model(model_path)

    train_datagen = generator(batch_size, img_size, test_list)
    predictions = model.predict(train_datagen)
    return predictions


def predict(i):
    global start, stop
    mask = np.argmax(predictions[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    image = PIL.ImageOps.autocontrast(array_to_img(mask))
    img = img_to_array(image)
    img = cv2.resize(img, input_size[::-1])
    mask = cv2.blur(img, (5, 5))

    nonzero = np.nonzero(mask)

    try:
        start = min(nonzero[0])
        stop = max(nonzero[0])
    except ValueError:
        print('no prediciton')

    y = np.linspace(start, stop, 10).astype(int)
    margin = 20
    leftx = np.zeros_like(y)
    rightx = np.zeros_like(y)

    for idx, val in enumerate(y):
        nonzerox = np.nonzero(mask[val, :])[0]
        if nonzerox.shape[0] == 0:
            continue
        leftx[idx] = nonzerox[0] + margin
        rightx[idx] = nonzerox[-1] - margin

    left_curve = np.polyfit(y, leftx, 2)
    right_curve = np.polyfit(y, rightx, 2)

    # points = np.zeros_like(mask)
    # for j in zip([leftx, rightx], [y, y]):
    #     a1, a2 = [k.reshape((-1, 1)) for k in j]
    #     con = np.concatenate((a1, a2), axis=1)
    #     for c in con:
    #         points = cv2.circle(points, c, 4, 1, -1)
    # cv2.imshow('points', points)
    # cv2.waitKey(0)

    # return left_curve, right_curve, mask, points

    return left_curve, right_curve, mask, stop


def display_prediction(i):

    test_image = cv2.imread(test_list[i])
    zeros = np.zeros_like(mask)
    poly = np.dstack((zeros, mask, zeros)).astype('uint8')
    prediction = cv2.addWeighted(test_image, 1, poly, 0.5, 0)
    out_img = visualise(prediction, left_curve, right_curve, start, stop)

    return prediction, out_img


def draw_circle(curve=None, color=(255, 0, 0)):
    if isinstance(curve, np.ndarray):
        circle = curve[0] * stop ** 2 + curve[1] * stop + curve[2]

    else:
        circle = curve

    cv2.circle(out_img, (int(circle), stop), 5, color, -1)
    return circle


# for train in ['train_3', 'train_4']:
#     predictions = choose_labels(train)
#     for i in range(len(test_list[:1])):
#         prediction, out_img = display_prediction(i)
#         cv2.imwrite(f'Pictures/{train}_prediction_{i}.jpg', prediction)
#         cv2.imshow('out_img', out_img)
#         cv2.waitKey(0)
#
#     for i in range(10, len(test_list[:11])):
#         prediction, out_img = display_prediction(i)
#         cv2.imwrite(f'Pictures/{train}_bad_fit_{i}.jpg', out_img)
#         cv2.imshow('out_img', out_img)
#         cv2.waitKey(0)
#
#     for i in range(19, len(test_list[:20])):
#         prediction, _ = display_prediction(i)
#         cv2.imwrite(f'Pictures/{train}_line_cross_{i}.jpg', prediction)
#         cv2.imshow('out_img', prediction)
#         cv2.waitKey(0)
#
#     for i in range(21, len(test_list[:22])):
#         prediction, _ = display_prediction(i)
#         cv2.imwrite(f'Pictures/{train}_adjacent_lane_{i}.jpg', prediction)
#         cv2.imshow('out_img', prediction)
#         cv2.waitKey(0)

i = 10
for i in range(len(test_list)):
    predictions = choose_labels('train_3')
    left_curve, right_curve, mask, stop = predict(i)

    prediction, out_img = display_prediction(i)
    left_curve = draw_circle(left_curve)
    right_curve = draw_circle(right_curve)
    center = out_img.shape[1] // 2
    middle  = left_curve + (right_curve - left_curve) // 2
    offset = middle - center

    draw_circle(middle, (0, 0, 255))
    draw_circle(center, (0, 255, 0))
    print(offset)

    cv2.imshow('out_img', out_img)
    cv2.waitKey(250)