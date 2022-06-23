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

batch_size = 32
img_size = (80, 160)
input_size = cv2.imread(test_list[0]).shape[:-1]

x = np.zeros((batch_size,) + img_size + (3,), dtype='float32')

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


train_datagen = generator(batch_size, img_size, test_list)
predictions = model.predict(train_datagen)


def display_mask(i):
    mask = np.argmax(predictions[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    image = PIL.ImageOps.autocontrast(array_to_img(mask))
    img = img_to_array(image)
    img = cv2.resize(img, input_size[::-1])
    blur = cv2.blur(img, (5, 5))

    zeros = np.zeros_like(blur)
    poly = np.dstack((zeros, blur, zeros)).astype('uint8')
    test = cv2.imread(test_list[i])
    output = cv2.addWeighted(test, 1, poly, 0.5, 0)
    cv2.imshow('output', output)
    cv2.waitKey(0)


# for i in range(1):
display_mask(479)

# for path in test_list[:10]:
#     test = cv2.imread(path)
#     width = test.shape[1]
#     height = test.shape[0]
#     image = cv2.resize(test, (160, 80)) / 255
#     image = image[None, ...]
#
#     prediction = model.predict(image)[0]
#     mask = np.argmax(prediction, axis=-1)
#     mask = np.expand_dims(mask, axis=-1)
#     img = PIL.ImageOps.autocontrast(array_to_img(mask))
#     # img.show()
#     img = img_to_array(img)
#     img = cv2.resize(img, (width, height))
#     blur = cv2.blur(img, (5, 5))
#
#     zeros = np.zeros_like(blur)
#     poly = np.dstack((zeros, blur, zeros)).astype('uint8')
#
#     output = cv2.addWeighted(test, 1, poly, 0.5, 0)
#     cv2.imshow('output', output)
#     cv2.waitKey(0)
#     #

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