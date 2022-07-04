from lane_detection3.lane_detection import visualise
import matplotlib.pyplot as plt
from tensorflow import keras
from imutils import paths
import numpy as np
import cv2
import os


def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            return os.path.join(path, file)


path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

dir_path = os.path.join(path, 'output')
test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

M = np.load('Pickles/M_video1.npy')
M_inv = np.load('Pickles/M_inv_video1.npy')

batch_size = 32
s_width = 160
s_height = 80
img_size = (s_height, s_width)
original_image = cv2.imread(test_list[0])
width = original_image.shape[1]
height = original_image.shape[0]


class generator(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, test_list, warp):
        self.batch_size = batch_size
        self.img_size = img_size
        self.test_list = test_list
        self.warp = warp

    def __len__(self):
        return len(self.test_list) // self.batch_size

    def __getitem__(self, idx):
        i = idx * batch_size
        test_batch = test_list[i: i + batch_size]
        x = np.zeros((batch_size,) + img_size + (3,), dtype='float32')

        for j, path in enumerate(test_batch):
            img = cv2.imread(path)
            if warp:
                img = cv2.resize(img, (width, width//2))
                img = cv2.warpPerspective(img, M, (width, width//2), flags=cv2.INTER_LINEAR)
            img = cv2.resize(img, img_size[::-1]) / 255
            x[j] = img

        return x


def choose_perspective(fname):
    global begin
    global warp

    warp = True
    begin = 0

    if fname == 'train_2':
        warp = False
        begin = 0.6 * height

    validation_path = os.path.join(dir_path, fname)
    model_path = find_file(validation_path, 'h5')
    model = keras.models.load_model(model_path)

    train_datagen = generator(batch_size, img_size, test_list, warp)
    predictions = model.predict(train_datagen)
    return predictions


def warp_perspective(image, M):
    width = image.shape[1]
    height = image.shape[0]
    resized = cv2.resize(image, (width, width // 2))
    warped = cv2.warpPerspective(resized, M, (width, width // 2), flags=cv2.INTER_LINEAR)
    out_img = cv2.resize(warped, (width, height))
    return out_img


def predict(i, perspective_mask):
    global start, stop

    points_arr = np.array(predictions[i] * width).astype(int).reshape((2, -1))
    y_range = np.linspace(begin, height - 1, 3).astype(int)
    nonzero = []

    radius = 6
    if perspective_mask:
        radius = 30

    mask = np.zeros((height, width))
    for arr in points_arr:
        side = np.zeros((height, width))
        for j in zip(arr, y_range):
            side = cv2.circle(side, (j), radius, 1, -1)

        # a1, a2 = [j.reshape((-1, 1)) for j in (arr, y_range)]
        # con = np.concatenate((a1, a2), axis=1)
        # lines = cv2.polylines(np.copy(side), [con], isClosed=False, color=1, thickness=warp*40+10)

        # resized = cv2.resize(lines, (width, height))

        if perspective_mask:
            side = warp_perspective(side, M_inv)

        mask += side


        nonzerox = side.nonzero()[1]
        nonzeroy = side.nonzero()[0]
        nonzero.append(nonzerox)
        nonzero.append(nonzeroy)

    mask = mask.astype('uint8') * 255

    try:
        start = min(min(nonzero[1]), min(nonzero[3]))
        stop = max(max(nonzero[1]), max(nonzero[3]))
    except ValueError:
        print('no prediciton')

    leftx, lefty, rightx, righty = nonzero
    left_curve = np.polyfit(lefty, leftx, 2)
    right_curve = np.polyfit(righty, rightx, 2)

    return left_curve, right_curve, mask


def make_mask(image, perspective_mask=False):
    global mask
    left_curve, right_curve, mask = predict(i, perspective_mask)
    zeros = np.zeros_like(mask)

    poly = np.dstack((zeros, mask, zeros))

    out_img = cv2.addWeighted(image, 1, poly, 0.5, 0)
    # out_img = visualise(out_img, left_curve, right_curve, start, stop)

    return out_img


def display_prediction(i):
    if warp:
        test_image = cv2.imread(test_list[i])
        out_img = make_mask(test_image, True)
        t_test_image = warp_perspective(test_image, M)
        t_out_img = make_mask(t_test_image)

        return [out_img, t_out_img]

    else:
        test_image = cv2.imread(test_list[i])
        out_img = make_mask(test_image)

        return [out_img]

    # cv2.imwrite(f'Pictures/{train}_out_img.jpg', out_img)
    # cv2.imshow('out_img', out_img)
    # cv2.waitKey(0)



for train in ('train_1', 'train_2'):
    predictions = choose_perspective(train)
    for i in range(len(test_list[:1])):
        print(i)
        out = display_prediction(i)
        if len(out) > 1:
            cv2.imwrite(f'Pictures/{train}_out_img.jpg', out[0])
            cv2.imwrite(f'Pictures/{train}_t_out_img.jpg', out[1])
            cv2.imshow('out_img', out[0])
            cv2.waitKey(0)
            cv2.imshow('t_out_img', out[1])
            cv2.waitKey(0)
        else:
            cv2.imwrite(f'Pictures/{train}_t_out_img.jpg', out[0])
            cv2.imshow('out_img', out[0])
            cv2.waitKey(0)


    for i in range(19, len(test_list[:20])):
        display_prediction(i)
        out = display_prediction(i)
        if len(out) > 1:
            cv2.imwrite(f'Pictures/{train}_line_cross.jpg', out[0])
            cv2.imwrite(f'Pictures/{train}_t_line_cross.jpg', out[1])
            cv2.imshow('out_img', out[0])
            cv2.waitKey(0)
            cv2.imshow('out_img', out[0])
            cv2.waitKey(0)
            cv2.imshow('t_out_img', out[1])
            cv2.waitKey(0)
        else:
            cv2.imwrite(f'Pictures/{train}_line_cross.jpg', out[0])
            cv2.imshow('out_img', out[0])
            cv2.waitKey(0)