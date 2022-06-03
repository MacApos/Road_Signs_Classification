import os
import cv2
import pickle
import shutil
import numpy as np
from imutils import paths
import PIL
from PIL import Image, ImageOps
from tensorflow import keras
from keras.utils.image_utils import load_img
from keras.utils import img_to_array, array_to_img


def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)


# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))
dir_path = os.path.join(path, 'output')
validation_path = [os.path.join(dir_path, folder) for folder in os.listdir(dir_path)][-1]
model_path = find_file(validation_path, 'h5')
print(model_path)
model = keras.models.load_model(model_path)

for test in test_list[:1]:
    image = Image.open(test).resize((320, 160))
    image.show()
    image = img_to_array(image)
    img = image[None, ...]

    val_preds = model.predict(img)[0]
    print(img[:-1])
    #
    # mask = np.argmax(val_preds[0], axis=-1)
    #
    # mask = np.expand_dims(mask, axis=-1)
    # # img = PIL.ImageOps.autocontrast(array_to_img(mask))
    # # img.show()
    # #
    cv2.imshow('mask', img[:-1])
    cv2.waitKey(0)

