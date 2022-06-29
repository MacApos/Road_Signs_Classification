from lane_detection3.lane_detection import im_show, fit_poly, visualise, generate_points
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow import keras
from imutils import paths
import numpy as np
import pickle
import cv2
import re
import os


def natural_keys(text):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', text)]


def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)


# path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, '../Output')
validation_path = os.path.join(dir_path, 'train_1')
# dir_list = os.listdir(dir_path)
# dir_list.sort(key=natural_keys)
# validation_path = [os.path.join(dir_path, folder) for folder in dir_list if folder.startswith('init')][-1]

test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

model_path = find_file(validation_path, 'h5')
model = keras.models.load_model(model_path)

M = np.load('../Pickles/M_video1.npy')
M_inv = np.load('../Pickles/M_inv_video1.npy')

batch_size = 32
s_width = 160
s_height = 80
img_size = (s_height, s_width)
original_image = cv2.imread(test_list[0])
width = original_image.shape[1]
height = original_image.shape[0]
y_range = np.linspace(0, s_height - 1, 3).astype(int)

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
            img = cv2.resize(img, (width, width//2))
            img = cv2.warpPerspective(img, M, (width, width//2), flags=cv2.INTER_LINEAR)
            img = cv2.resize(img, img_size[::-1]) / 255
            x[j] = img

        return x


train_datagen = generator(batch_size, img_size, test_list)
predictions = model.predict(train_datagen)

i = 0
points_arr = np.array(predictions[i] * s_width).astype(int).reshape((2, -1))

coefficients = []
points_nonzero = []
lines_nonzero = []

for arr in points_arr:
    side = np.zeros((s_height, s_width))
    points = np.copy(side)
    coefficients.append(np.polyfit(y_range, arr, 2))
    for j in zip(arr, y_range):
        cv2.circle(points, (j), 4, (255, 0, 0), -1)

    a1, a2 = [j.reshape((-1, 1)) for j in (arr, y_range)]
    con = np.concatenate((a1, a2), axis=1)
    lines = cv2.polylines(np.copy(side), [con], isClosed=False, color=1, thickness=5)

    for nonzero, image in zip([points_nonzero, lines_nonzero], [points, lines]):
        image = cv2.resize(image, (width, width//2))
        warp = cv2.warpPerspective(image, M_inv, (width, width//2), flags=cv2.INTER_LINEAR)
        resized = cv2.resize(warp, (width, height))

        nonzerox = resized.nonzero()[1]
        nonzeroy = resized.nonzero()[0]
        nonzero.append(nonzerox)
        nonzero.append(nonzeroy)

output = []

for nonzero in points_nonzero, lines_nonzero:
    leftx, lefty, rightx, righty = nonzero
    start = min(min(lefty), min(righty))
    left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
    zeros = np.zeros((height, width))
    visualization = visualise(zeros, left_curve, right_curve, start, show_lines=True)
    output.append(visualization)

zeros = np.zeros((height, width))
visualization = visualise(np.zeros((height, width)), coefficients[0], coefficients[1], start, show_lines=True)
output.insert(0, visualization)

plt.figure(figsize=(16, 8))
titles = ['coefficents', 'points', 'lines']
for idx, img in enumerate(output):
    plt.subplot(1, len(output), idx+1)
    plt.title(titles[idx])
    plt.grid(False)
    plt.axis(False)
    plt.imshow(img)
plt.show()
