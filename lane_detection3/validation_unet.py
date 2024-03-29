import os
import cv2
import pickle
import shutil
import numpy as np
from imutils import paths
import PIL
from PIL import ImageOps
from tensorflow import keras
from lane_detection import visualise
# from keras.utils import img_to_array, array_to_img
from keras.preprocessing.image import img_to_array, array_to_img


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



def choose_labels(fname):
    validation_path = os.path.join(dir_path, fname)
    model_path = find_file(validation_path, 'h5')
    model = keras.models.load_model(model_path)
    print(model_path)

    train_datagen = generator(batch_size, img_size, test_list)
    predictions = model.predict(train_datagen)
    return predictions


def predict(i):
    global start, stop
    mask = np.argmax(predictions[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    image = PIL.ImageOps.autocontrast(array_to_img(mask))
    img = img_to_array(image)
    img = cv2.resize(img, input_size[::-1])
    mask = cv2.blur(img, (5, 5))

    nonzero = np.nonzero(mask)

    try:
        start = min(nonzero[0])
        stop = max(nonzero[0])
    except ValueError:
        print('no prediciton')

    y = np.linspace(start, stop, 10).astype(int)
    margin = 20
    leftx = np.zeros_like(y)
    rightx = np.zeros_like(y)

    for idx, val in enumerate(y):
        nonzerox = np.nonzero(mask[val, :])[0]
        if nonzerox.shape[0] == 0:
            continue
        leftx[idx] = nonzerox[0] + margin
        rightx[idx] = nonzerox[-1] - margin

    left_curve = np.polyfit(y, leftx, 2)
    right_curve = np.polyfit(y, rightx, 2)

    # points = np.zeros_like(mask)
    # for j in zip([leftx, rightx], [y, y]):
    #     a1, a2 = [k.reshape((-1, 1)) for k in j]
    #     con = np.concatenate((a1, a2), axis=1)
    #     for c in con:
    #         points = cv2.circle(points, c, 4, 1, -1)
    # cv2.imshow('points', points)
    # cv2.waitKey(0)

    # return left_curve, right_curve, mask, points

    return left_curve, right_curve, mask, stop


def display_prediction(i):
    test_image = cv2.imread(test_list[i])
    zeros = np.zeros_like(mask)
    poly = np.dstack((zeros, mask, zeros)).astype('uint8')
    prediction = cv2.addWeighted(test_image, 1, poly, 0.5, 0)
    out_img = visualise(test_image, left_curve, right_curve, start, stop)

    return prediction, out_img


def draw_circle(curve=None, color=(255, 0, 0)):
    if isinstance(curve, np.ndarray):
        circle = curve[0] * stop ** 2 + curve[1] * stop + curve[2]

    else:
        circle = curve
        cv2.circle(out_img, (int(circle), stop), 5, color, -1)

    return circle


# for train in ['train_3', 'train_4']:
#     predictions = choose_labels(train)
#     indices = [0, 10, 19, 21]
#     filename = ['prediction', 'bad_fit', 'line_cross', 'adjacent_lane']
#
#     for idx, i in enumerate(indices):
#         left_curve, right_curve, mask, stop = predict(i)
#         prediction, out_img = display_prediction(i)
#         cv2.imwrite(f'Pictures/validation/{train}_{filename[idx]}_{i}.jpg', prediction)
#         cv2.imshow('out_img', out_img)
#         cv2.waitKey(0)

predictions = choose_labels('train_3')
mean_width = 485
m_per_px = 3.7 / 485
image = cv2.imread(test_list[0])
center = image.shape[1] // 2

offset_text = 'Offset: 0.00 m'
cross_text = 'Line cross'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 0, 0)
thickness = 2
(offset_width, offset_height), _ = cv2.getTextSize(offset_text, font, font_scale, thickness)
offset_x = center - offset_width // 2
offset_y = 50 + offset_height // 2

(cross_width, cross_height), _ = cv2.getTextSize(cross_text, font, font_scale, thickness)
radius = 10
space = 5
circle_x = center - (cross_width + radius*2 + space)// 2
circle_y = offset_y + int(offset_height * 1.5)

cross_x = circle_x + radius + space
cross_y = offset_y + int(offset_height * 2)

j = 0
for i in range(len(test_list)):
    left_curve, right_curve, mask, stop = predict(i)
    prediction, out_img = display_prediction(i)

    if i >210:
        cv2.imshow(f'out_img_{i}', prediction)
        while True:
            k = cv2.waitKey(0) & 0xFF
            print(k)
            if k == 27:
                cv2.destroyAllWindows()
                break

    # left_stop = draw_circle(left_curve)
    # right_stop = draw_circle(right_curve)
    #
    # width = right_stop - left_stop
    #
    # middle  = left_stop + width // 2
    # offset = (middle - center) * m_per_px
    #
    # print(offset)
    # draw_circle(middle, (0, 0, 255))
    # draw_circle(center, (0, 255, 0))
    #
    # offset_text = 'Offset: {:.2f} m'.format(offset)
    # cv2.putText(out_img, offset_text, (offset_x, offset_y), font, font_scale, color, thickness, cv2.LINE_AA)
    #
    # if i == 15:
    #     cv2.imwrite(f'Pictures/validation/offset_{i}.jpg', out_img)
    #
    # if width < mean_width / 3:
    #     j += 1
    #     if j == 2:
    #         cv2.circle(out_img, (circle_x, circle_y), radius, (0, 0, 255), -1)
    #         cv2.putText(out_img, cross_text, (cross_x, cross_y), font, font_scale, color, thickness, cv2.LINE_AA)
    #         cv2.imwrite(f'Pictures/validation/line_cross_{i}.jpg', out_img)
    #         break
    #
    # cv2.imshow(f'out_img_{i}', out_img)
    # cv2.waitKey(0)


