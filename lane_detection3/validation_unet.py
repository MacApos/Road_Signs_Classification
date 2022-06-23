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

dir_path = os.path.join(path, 'output')
# validation_path = [os.path.join(dir_path, folder) for folder in os.listdir(dir_path)][-1]
validation_path = os.path.join(dir_path, 'initialized_3')

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


def create_mask(i):
    mask = np.argmax(predictions[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    image = PIL.ImageOps.autocontrast(array_to_img(mask))
    img = img_to_array(image)
    img = cv2.resize(img, input_size[::-1])
    mask = cv2.blur(img, (5, 5))
    image = cv2.imread(test_list[i])
    return mask, image


def display_mask(mask, image):
    zeros = np.zeros_like(mask)
    poly = np.dstack((zeros, mask, zeros)).astype('uint8')
    return cv2.addWeighted(image, 1, poly, 0.5, 0)

output = []
for i in range(60):
    mask, image = create_mask(i)
    mask = display_mask(mask, image)
    if i < 3:
        cv2.imwrite(os.path.join(dir_path, f'unet{i}.jpg'), mask)
    img = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    output.append(img)

save_path = os.path.join(dir_path, 'unet.gif')
print(save_path)

import imageio
with imageio.get_writer(save_path, mode='I', fps=3) as writer:
    for image in output:
        print('saving')
        writer.append_data(image)