import os
import cv2
import random
import numpy as np
from imutils import paths
import shutil


def fill(img, w, h):
    return cv2.resize(img, (w, h), cv2.INTER_CUBIC)


def fill_nearest(img, top, bottom, left, right):
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)


def horizontal_shift(image, shift):
    h, w = image.shape[:2]
    random_shift = random.uniform(-shift, shift)
    w_shift = int(w * random_shift)

    if random_shift >= 0:
        shifted_img = image[:, w_shift:, :]
        shifted_img = fill_nearest(shifted_img, 0, 0, w_shift, 0)

    elif random_shift < 0:
        shifted_img = image[:, int(-1 * w_shift):, :]
        shifted_img = fill_nearest(shifted_img, 0, 0, 0, -1 * w_shift)

    return shifted_img

def vertical_shift(image, shift):
    h, w = image.shape[:2]
    random_shift = random.uniform(-shift, shift)
    v_shift = int(h * random_shift)

    if random_shift >= 0:
        shifted_img = image[int(v_shift):, :, :]
        shifted_img = fill_nearest(shifted_img, v_shift, 0, 0, 0)

    elif random_shift < 0:
        shifted_img = image[int(-1 * v_shift):, :, :]
        shifted_img = fill_nearest(shifted_img, 0, -1 * v_shift, 0, 0)

    return shifted_img

def zoom(image, zoom):
    h, w = image.shape[:2]
    random_zoom = random.uniform(0, zoom)

    h_zoom = int(h * random_zoom)
    w_zoom = int(w * random_zoom)
    zoomed_img = image[h_zoom:h-h_zoom, w_zoom:w-w_zoom, :]
    zoomed_img = fill(zoomed_img, w, h)
    # zoomed_img = cv2.copyMakeBorder(zoomed_img, h_zoom, h_zoom, w_zoom, w_zoom, cv2.BORDER_CONSTANT, 0)
    # cv2.rectangle(img, (w_zoom, h_zoom), (w-w_zoom, h-h_zoom), (0, 255, 0), 4)

    return zoomed_img

def rotate(image, angle):
    h, w = image.shape[:2]
    angle_range = [i for i in range(-angle, angle, 1) if i!= 0]
    random_angle = random.sample(angle_range, 1)[0]

    M = cv2.getRotationMatrix2D((w // 2, h // 2), random_angle, 1)
    rotated_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.INTER_LINEAR)

    return rotated_img


def flip(image):
    flipped_img = np.fliplr(image)

    return flipped_img


def augmentation(image):
    h, w = image.shape[:2]

    img = horizontal_shift(image, 0.1)
    img = vertical_shift(img, 0.1)
    # img = zoom(img, 0.2)
    # img = rotate(img, 4)
    img = flip(img)

    return img


def prepare(path):
    data_path = os.path.join(path, 'train')
    aug_path = os.path.join(path, 'augmentation')

    data_list = list(paths.list_images(data_path))


    print('Delete previous data?', ' [y/n]')
    x = input()

    if os.path.exists(aug_path) and x == 'y':
        # if os.path.exists(folder_path):
        shutil.rmtree(aug_path)

    if not os.path.exists(aug_path):
        os.mkdir(aug_path)

    for path in data_list[:100]:
        image = cv2.imread(path)
        basename = os.path.basename(path)
        save_path = os.path.join(aug_path, basename)

        augemnted_img = augmentation(image)

        if not os.path.exists(save_path):
            print(f'saving {basename}')
            cv2.imwrite(save_path, augemnted_img)


# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'









