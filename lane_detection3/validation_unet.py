import os
import cv2
import pickle
import shutil
import numpy as np
from imutils import paths
import PIL
from PIL import Image, ImageOps
from tensorflow import keras
# from keras.utils import img_to_array, array_to_img
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img


def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)


path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

test_path = os.path.join(path, 'test')
dir_path = os.path.join(path, 'output')
# validation_path = [os.path.join(dir_path, folder) for folder in os.listdir(dir_path)][-1]
validation_path = os.path.join(dir_path, 'initialized_3')
model_path = find_file(validation_path, 'h5')

test_list = list(paths.list_images(test_path))
model = keras.models.load_model(model_path)


for path in test_list[:1]:
    test = cv2.imread(path)
    width = test.shape[1]
    height = test.shape[0]
    image = cv2.resize(test, (160, 80)) / 255
    image = image[None, ...]

    prediction = model.predict(image)[0]
    mask = np.argmax(prediction, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(array_to_img(mask))
    # img.show()
    img = img_to_array(img)
    img = cv2.resize(img, (width, height))
    blur = cv2.blur(img, (5, 5))

    zeros = np.zeros_like(blur)
    poly = np.dstack((zeros, blur, zeros)).astype('uint8')

    output = cv2.addWeighted(test, 1, poly, 0.5, 0)
    cv2.imshow('output', output)
    cv2.waitKey(0)
    #

# test_path = os.path.join(path, 'test')
# test_list = list(paths.list_images(test_path))
# test_pickle = 'Pickles/test.p'
# print(img_size[1], img_size[0])
# if not os.path.exists(test_pickle):
#     test = [cv2.resize(cv2.imread(path), (img_size[1], img_size[0])) for path in test_list]
#     pickle.dump(test, open(test_pickle, 'wb'))
# else:
#     test = pickle.load(open(test_pickle, 'rb' ))
#
# # valid = x_test[0]
# valid = cv2.imread(r'C:\Nowy folder\10\Praca\Datasets\Video_data\test\00000.jpg')
# valid = cv2.resize(valid, (160, 80)) / 255
# cv2.imshow('valid', valid)
# cv2.waitKey(0)
# valid = valid[None, ...]
# print(valid.shape)
#
# val_preds = model.predict(valid)[0]
# mask = np.argmax(val_preds, axis=-1)
# mask = np.expand_dims(mask, axis=-1)
# img = PIL.ImageOps.autocontrast(array_to_img(mask))
# img = img_to_array(img)
# cv2.imshow('predictions', img)
# cv2.waitKey(0)
#
# # for image in [test[-2]]:
# #     cv2.imshow('image', image)
# #     cv2.waitKey(500)
# #     image = image[None, ...]
# #     print(image.shape)
# #
# #     val_preds = model.predict(image)[0]
# #
# #     mask = np.argmax(val_preds, axis=-1)
# #     mask = np.expand_dims(mask, axis=-1)
# #     img = PIL.ImageOps.autocontrast(array_to_img(mask))
# #     img = img_to_array(img)
# #     cv2.imshow('predictions', img)
# #     cv2.waitKey(0)