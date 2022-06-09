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
validation_path = os.path.join(dir_path, 'initialized_1(1)')
model_path = find_file(validation_path, 'h5')

test_list = list(paths.list_images(test_path))
model = keras.models.load_model(model_path)


for path in test_list[250:]:
    test = cv2.imread(path)
    image = cv2.resize(test, (160,80))
    cv2.imshow('image', image)
    cv2.waitKey(0)
    image = image[None, ...]

    prediction = model.predict(image)[0]
    mask = np.argmax(prediction, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(array_to_img(mask))
    # img.show()
    img = img_to_array(img)
    cv2.imshow('img', img)
    cv2.waitKey(500)
