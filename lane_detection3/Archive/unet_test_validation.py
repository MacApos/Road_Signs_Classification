import os
import cv2
import pickle
import shutil
import numpy as np
from imutils import paths
import PIL
from PIL import Image, ImageOps
from tensorflow import keras
from keras.preprocessing.image import img_to_array, array_to_img

path = r'C:\Nowy folder\10\Praca\Datasets\unet'
test_path = os.path.join(path, 'test')
output_path = os.path.join(path, 'output')
model_path = os.path.join(output_path, 'colab_train.h5')

valid_data_list = pickle.load(open('Pickles/valid_data_list.p', 'rb' ))
if not len(os.listdir(test_path)):
    for path in valid_data_list:
        dst = os.path.join(test_path, os.path.basename(path))
        shutil.copyfile(path, dst)

test_list = list(paths.list_images(test_path))
model = keras.models.load_model(model_path)

for test in test_list[:1]:
    image = Image.open(test).resize((160,160))
    image = img_to_array(image)
    img = image[None, ...]

    val_preds = model.predict(img)

    mask = np.argmax(val_preds[0], axis=-1)

    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(array_to_img(mask))
    img.show()

    # cv2.imshow('mask', img_to_array(img))
    # cv2.waitKey(0)

