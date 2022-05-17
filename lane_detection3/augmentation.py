import os
import cv2
import random
import numpy as np
from imutils import paths
import shutil

def sort_path(path):
    sorted_path = []
    for file in os.listdir(path):
        number = int(''.join(n for n in file if n.isdigit()))
        sorted_path.append(number)

    sorted_path = sorted(sorted_path)
    return [path + fr'\{str(f)}.jpg' for f in sorted_path]


def resize(path):
    i = 0
    for image in image_list(path):
        img = cv2.imread(image)

        if height == 720:
            resized_img = img[260:, :]

            # scaled_img = cv2.resize(image, (width//8, height//8))

            cv2.imwrite(image, resized_img)
            print('resize', i, image)
            i += 1


def fill(img, w, h):
    return cv2.resize(img, (w, h), cv2.INTER_CUBIC)

def fill_nearest(img, top, bottom, left, right):
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)


def save(fname, image_path, image):
    file_path = os.path.join(aug_path, fname)
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    cv2.imwrite(image_path, image)
    cv2.imwrite(file_path + fr'\{os.path.basename(image_path)}', image)


def horizontal_shift(path, shift, ratio=0.1):
    i = 0
    for image in random_list(path, ratio):
        img = cv2.imread(image)

        random_shift = random.uniform(-shift, shift)
        h, w = img.shape[:2]
        w_shift = int(w * random_shift)

        if random_shift >= 0:
            shifted_img = img[:, w_shift:, :]
            shifted_img = fill_nearest(shifted_img, 0, 0, w_shift, 0)

        elif random_shift < 0:
            shifted_img = img[:, int(-1 * w_shift):, :]
            shifted_img = fill_nearest(shifted_img, 0, 0, 0, -1 * w_shift)

        save('h_shift', image, shifted_img)
        # cv2.imwrite(image, shifted_img)
        print(i, 'h_shift', image)
        i += 1


def vertical_shift(path, shift, ratio=0.1):
    i = 0
    for image in random_list(path, ratio):
        img = cv2.imread(image)

        random_shift = random.uniform(-shift, shift)
        h, w = img.shape[:2]
        v_shift = int(h * random_shift)

        if random_shift >= 0:
            shifted_img = img[int(v_shift):, :, :]
            shifted_img = fill_nearest(shifted_img, v_shift, 0, 0, 0)

        elif random_shift < 0:
            shifted_img = img[int(-1 * v_shift):, :, :]
            shifted_img = fill_nearest(shifted_img, 0, -1 * v_shift, 0, 0)

        save('v_shift', image, shifted_img)
        # cv2.imwrite(image, shifted_img)
        print(i, 'v_shift', image)
        i += 1


def zoom(path, zoom, ratio):
    i = 0
    for image in random_list(path, ratio):
        img = cv2.imread(image)

        random_zoom = random.uniform(0, zoom)
        w = img.shape[1]
        h = img.shape[0]

        h_zoom = int(h* random_zoom)
        w_zoom = int(w * random_zoom)
        zoomed_img = img[h_zoom:h-h_zoom, w_zoom:w-w_zoom, :]
        zoomed_img = fill(zoomed_img, w, h)
        # zoomed_img = cv2.copyMakeBorder(zoomed_img, h_zoom, h_zoom, w_zoom, w_zoom, cv2.BORDER_CONSTANT, 0)
        # cv2.rectangle(img, (w_zoom, h_zoom), (w-w_zoom, h-h_zoom), (0, 255, 0), 4)

        save('zoom', image, zoomed_img)
        # cv2.imwrite(image, zoomed_img)
        print(i, 'zoom', image)
        i += 1

def rotate(path, angle, ratio):
    # i = len(os.listdir(path))
    i = 0
    angle_range = [i for i in range(-angle, angle, 1) if i!= 0]
    for image in random_list(path, ratio):
        img = cv2.imread(image)

        random_angle = random.sample(angle_range, 1)[0]
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), random_angle, 1)
        rotated_img = cv2.warpAffine(img, M, (w, h))

        save('rotation', image, rotated_img)
        # cv2.imwrite(image, flipped_img)
        print(i, 'rotation', image)
        i += 1

        # cv2.imwrite(path+fr'\{i:05d}.jpg', rotated_img)
        # print(i, path+fr'\{i:05d}.jpg')


def flip(path, ratio):
    i = 0
    for image in random_list(path, ratio):
        img = cv2.imread(image)

        flipped_img = np.fliplr(img)

        save('flip', image, flipped_img)
        # cv2.imwrite(image, flipped_img)
        print(i, 'flip', image)
        i += 1


def random_list(path, range):
    img_list = image_list(path)
    return random.sample(img_list,  int(len(os.listdir(path)) * range))


def image_list(path):
    return list(paths.list_images(path))

# path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
aug_path = os.path.join(path, 'augmentation')
raw_data_path = os.path.join(path, 'raw_data')
data_path = os.path.join(path, 'data')

print('commit')

if os.path.exists(data_path):
    if len(os.listdir(raw_data_path)) != len(os.listdir(data_path)):
        shutil.rmtree(data_path)

if not os.path.exists(data_path):
    shutil.copytree(raw_data_path, data_path)

# if os.path.exists(aug_path):
#     shutil.rmtree(aug_path)

if not os.path.exists(aug_path):
    os.mkdir(aug_path)


data_list = image_list(data_path)

image = cv2.imread(data_list[0])
height = image.shape[0]
width = image.shape[1]

# resize(data_path)

horizontal_shift(data_path, 0.1, 0.1)
vertical_shift(data_path, 0.1, 0.1)
zoom(data_path, 0.1, 0.1)
rotate(data_path, 3, 0.1)
flip(data_path, 0.1)
