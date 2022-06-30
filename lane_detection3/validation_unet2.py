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
from keras.preprocessing.image import img_to_array, array_to_img


def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)


# path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, 'output')
# validation_path = [os.path.join(dir_path, folder) for folder in os.listdir(dir_path)][-1]
validation_path = os.path.join(dir_path, 'train_3')

test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

model_path = find_file(validation_path, 'h5')
model = keras.models.load_model(model_path)

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


train_datagen = generator(batch_size, img_size, test_list)
predictions = model.predict(train_datagen)


def predict(i):
    global start, stop

    mask = np.argmax(predictions[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    image = PIL.ImageOps.autocontrast(array_to_img(mask))
    img = img_to_array(image)
    img = cv2.resize(img, input_size[::-1])
    blur = cv2.blur(img, (5, 5))

    # zeros = np.zeros_like(blur)
    # poly = np.dstack((zeros, mask, zeros)).astype('uint8')
    # test_image = cv2.imread(test_list[i])
    # prediction = cv2.addWeighted(image, 1, poly, 0.5, 0)

    nonzero = np.nonzero(blur)
    start = min(nonzero[0])
    stop = max(nonzero[0])
    y = np.linspace(start, stop, 16).astype(int)

    leftx = np.zeros_like(y)
    rightx = np.zeros_like(y)

    for idx, val in enumerate(y):
        nonzerox = np.nonzero(blur[val, :])[0]
        if nonzerox.shape[0] == 0:
            continue
        leftx[idx] = nonzerox[0]
        rightx[idx] = nonzerox[-1]

    left_curve = np.polyfit(y, leftx, 2)
    right_curve = np.polyfit(y, rightx, 2)

    return left_curve, right_curve


from lane_detection3.lane_detection import visualise
for i in range(len(test_list)):
    test_image = cv2.imread(test_list[i])
    left_curve, right_curve = predict(i)
    out_img = visualise(test_image, left_curve, right_curve, start, stop)
    cv2.imshow('out_img', out_img)
    cv2.waitKey(0)
