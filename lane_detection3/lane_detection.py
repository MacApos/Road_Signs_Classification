import os
import cv2
import pickle
import shutil
import numpy as np
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt


def im_show(image, name='Image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)


def to_csv(name, array):
    array_list = []
    for arr in array:
        if isinstance(arr, np.ndarray):
            arr = arr[0]
        arr = str(arr).replace('.', ',')
        array_list.append(arr)

    df = pd.DataFrame(array_list)
    # path = os.path.join(r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3\Arrays', name)
    path = os.path.join(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Arrays', name)
    df.to_csv(path, sep='\t', index=False, header=False)


def warp_perspective(image, from_, to, width, height):
    M = cv2.getPerspectiveTransform(from_, to)
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warp, M


def threshold(image, T):
    _, image = cv2.threshold(image, T, 250, cv2.THRESH_BINARY)
    return image


def color_mask(img, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary_output = np.zeros_like(gray)
    binary_output[(gray > thresh[0]) & (gray <= thresh[1])] = 255
    return binary_output


def gray_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def draw_lines(image, arr, point_color=(255, 0, 0), line_color=(0, 255, 0)):
    arr = arr.astype(int)
    copy = np.copy(image)
    for i in range(arr.shape[0]):
        x, y = arr[i][0], arr[i][1]
        x_0, y_0 = arr[i - 1][0], arr[i - 1][1]
        cv2.circle(copy, (x, y), radius=1, color=point_color, thickness=10)
        cv2.line(copy, (x, y), (x_0, y_0), color=line_color, thickness=4)
    return copy


def to_jpg(name, image):
    path = 'Pictures/' + name + '.jpg'
    cv2.imwrite(path, image)


def prepare(image, src, dst):
    width = image.shape[1]
    height = image.shape[0]
    box = draw_lines(image, src)
    box = draw_lines(box, dst, line_color=(0, 0, 255))
    warp, _ = warp_perspective(image, src, dst, width, height)
    gray = gray_img(warp)
    max_val = max(np.amax(gray, axis=1)).astype(int)
    thresh = color_mask(warp, (max_val * 0.65, max_val))

    return warp, thresh


def find_single_lane(side_current, count):
    side_left = side_current - margin
    side_right = side_current + margin
    cv2.rectangle(out_img, (side_left, low), (side_right, high), (0, 255, 0), 4)

    side_nonzero = ((nonzeroy >= low) & (nonzeroy <= high) &
                    (nonzerox >= side_left) & (nonzerox <= side_right)).nonzero()[0]

    side_indicator = True

    if (side_left < 0 or side_right > width) and len(side_nonzero) == 0:
        count += 1
    else:
        count = 0

    # 35//2
    if count >= 17:
        side_indicator = False

    if len(side_nonzero) > minpix:
        side_current = int(np.mean(nonzerox[side_nonzero]))

    return side_current, side_nonzero, side_indicator, count, side_left, side_right


def find_lanes(image):
    global margin, minpix
    global out_img
    global low, high
    global nonzerox, nonzeroy
    global left_intercept, left_slope, right_intercept, right_slope
    global lefty, leftx, righty, rightx

    number = 35
    minpix = 50
    margin = 100

    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((image, image, image))

    midpoint = int(histogram.shape[0] // 2)
    left = np.argmax(histogram[:midpoint])
    right = midpoint + np.argmax(histogram[midpoint:])

    if np.argmax(histogram[:midpoint]) == 0:
        left = 0 + margin
    if np.argmax(histogram[midpoint:]) == 0:
        right = width - margin

    # if left - margin<= 0:
    #     left = 0 + margin
    # if right + margin >= width:
    #     right = width - margin

    left_current = left
    right_current = right

    left_indicator = True
    right_indicator = True

    left_idx = []
    right_idx = []

    nonzero = np.nonzero(image)
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_count = 0
    right_count = 0

    win_height = int(height // number)

    for i in range(number):
        low = image.shape[0] - win_height * (i + 1)
        high = image.shape[0] - win_height * i

        if left_indicator and right_indicator:
            left_current, left_nonzero, left_indicator, left_count, _, left_right = find_single_lane(left_current,
                                                                                                     left_count)
            right_current, right_nonzero, right_indicator, right_count, right_left, _ = find_single_lane(right_current,
                                                                                                         right_count)
            left_idx.append(left_nonzero)
            right_idx.append(right_nonzero)

        elif left_indicator:
            left_current, left_nonzero, left_indicator, left_count, _, _ = find_single_lane(left_current, left_count)
            left_idx.append(left_nonzero)

        elif right_indicator:
            right_current, right_nonzero, right_indicator, right_count, _, _ = find_single_lane(right_current,
                                                                                                right_count)
            right_idx.append(right_nonzero)

        else:
            print('break')
            break

    try:
        left_idx = np.concatenate(left_idx)
        right_idx = np.concatenate(right_idx)
    except AttributeError:
        pass

    leftx1 = nonzerox[left_idx]
    lefty1 = nonzeroy[left_idx]
    rightx1 = nonzerox[right_idx]
    righty1 = nonzeroy[right_idx]

    if (len(leftx1) == 0 or len(leftx1) <= minpix) and (len(rightx1) == 0 or len(rightx1) <= minpix):
        print('no right and no left')
        leftx1 = previous_frame[0][0]
        lefty1 = previous_frame[0][1]
        rightx1 = previous_frame[0][2]
        righty1 = previous_frame[0][3]

    elif len(leftx1) <= minpix:
        print('no left')
        # leftx1 = rightx1 - width // 2
        leftx1 = width - rightx1
        lefty1 = righty1

    elif len(rightx1) <= minpix:
        print('no right')
        # rightx1 = leftx1 + width // 2
        rightx1 = width - leftx1
        righty1 = lefty1

    # print(len(leftx1), len(rightx1))

    left_curve1 = np.polyfit(lefty1, leftx1, 2)
    right_curve1 = np.polyfit(righty1, rightx1, 2)

    left_nonzero1 = (
            (nonzerox > (left_curve1[0] * (nonzeroy ** 2) + left_curve1[1] * nonzeroy + left_curve1[2] - margin)) &
            (nonzerox < (left_curve1[0] * (nonzeroy ** 2) + left_curve1[1] * nonzeroy + left_curve1[2] + margin)))

    right_nonzero1 = (
            (nonzerox > (right_curve1[0] * (nonzeroy ** 2) + right_curve1[1] * nonzeroy + right_curve1[2] - margin)) &
            (nonzerox < (right_curve1[0] * (nonzeroy ** 2) + right_curve1[1] * nonzeroy + right_curve1[2] + margin)))

    leftx = nonzerox[left_nonzero1]
    lefty = nonzeroy[left_nonzero1]
    rightx = nonzerox[right_nonzero1]
    righty = nonzeroy[right_nonzero1]

    if len(leftx) <= minpix or len(rightx) <= minpix:
        leftx, lefty, rightx, righty = leftx1, lefty1, rightx1, righty1

    return leftx, lefty, rightx, righty, out_img


def fit_poly(leftx, lefty, rightx, righty):
    left_curve = np.polyfit(lefty, leftx, 2)
    right_curve = np.polyfit(righty, rightx, 2)

    return left_curve, right_curve

def generate_points(image, left_curve, right_curve, start=0):
    height = image.shape[0]
    y = np.linspace(start, height - 1, 15).astype(int).reshape((-1, 1))
    fit_left = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
    fit_right = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]

    empty = []
    flipud = False

    for arr in fit_left, fit_right:
        arr = arr.astype(int)
        con = np.concatenate((arr, y), axis=1)

        if flipud:
            con = np.flipud(con)

        flipud = True
        empty.append(con)

    points = np.array(empty)

    return points


def visualise(image, left_curve, right_curve, start=0, show_lines=False, show_points=False):
    points_arr = generate_points(image, left_curve, right_curve, start)
    colors = [(0, 0, 255), (255, 0, 0)]

    for idx, arr in enumerate(points_arr):
        if show_lines:
            cv2.polylines(image, [arr], isClosed=False, color=colors[idx], thickness=4)

        if show_points:
            for point in arr:
                cv2.circle(image, tuple(point), 5, colors[idx], -1)

    return image


def scale_and_perspective(image, left_curve, right_curve, src, dst, scale_factor, perspective=False):
    width = image.shape[1]
    height = image.shape[0]
    scaled_width = int(width * scale_factor)
    scaled_height = int(scaled_width / 2)

    M_inv = cv2.getPerspectiveTransform(dst, src)
    points_arr = generate_points(image, left_curve, right_curve)

    nonzero = []

    for arr in points_arr:
        side = np.zeros((height, width))
        side = cv2.polylines(side, [arr], isClosed=False, color=1, thickness=20)
        if perspective:
            side = cv2.warpPerspective(side, M_inv, (width, height), flags=cv2.INTER_LINEAR)
        side = cv2.resize(side, (scaled_width, scaled_height))
        nonzerox = side.nonzero()[1]
        nonzeroy = side.nonzero()[0]
        nonzero.append(nonzerox)
        nonzero.append(nonzeroy)

    leftx, lefty, rightx, righty = nonzero

    if len(leftx) == 0:
        leftx = width - rightx
        lefty = righty

    if len(rightx) == 0:
        rightx = width - leftx
        righty = lefty

    return leftx, lefty, rightx, righty


def visualise_perspective(image, left_curve, right_curve, src, dst, scale_factor):
    poly = np.zeros_like(image)
    width = poly.shape[1]
    height = poly.shape[0]
    scaled_width = int(width * scale_factor)
    scaled_height = int(scaled_width / 2)

    M_inv = cv2.getPerspectiveTransform(dst, src)
    points_arr = generate_points(image, left_curve, right_curve)

    points = np.vstack((points_arr[0], points_arr[1]))
    cv2.fillPoly(poly, [points], (0, 255, 0))
    poly = cv2.warpPerspective(poly, M_inv, (width, height), flags=cv2.INTER_LINEAR)
    out_frame = cv2.addWeighted(image, 1, poly, 0.5, 0)
    poly = poly[:, :, 1]

    poly = cv2.resize(poly, (scaled_width, scaled_height))
    out_frame = cv2.resize(out_frame, (scaled_width, scaled_height))

    return poly, out_frame


def params():
    width = 1280
    height = 480
    video1 = {'name': 'video1',
              'src': np.float32([[290, 410], [550, 285]]),
              'thresh': 0.65,
              'limit': 2548}

    video2 = {'name': 'video2',
              'src': np.float32([[285, 410], [550, 285]]),  # [[:, 390+20], [:, 265+20]]
              'thresh': 0.55,
              'limit': 4122}

    video3 = {'name': 'video3',
              'src': np.float32([[280, 420], [570, 250]]),
              'thresh': 0.85,
              'limit': 2833}

    video4 = {'name': 'video4',
              'src': np.float32([[270, 420], [550, 265]]),
              'thresh': 0.9,
              'limit': 1840}

    video_list = [video1]

    for video in video_list:
        template = video['src']
        src = np.float32([[template[0][0], template[0][1]],
                          [template[1][0], template[1][1]],
                          [width - template[1][0], template[1][1]],
                          [width - template[0][0], template[0][1]]])

        warp_arr = {'src': src}
        video.update(warp_arr)

    dst = np.float32([[0, height],
                      [0, 0],
                      [width, 0],
                      [width, height]])

    return video_list, dst

def make_input(message):
    print(message, ' [y/n]')
    x = input()
    x = x.lower()
    if x != 'y' and x != 'n':
        raise Exception('Invalid input')

    return x

def detect_lines(path):
    global width, height
    global previous_frame
    global M_inv

    root_path = os.path.dirname(__file__)
    data_path = os.path.join(path, 'train')
    frames_path = os.path.join(path, 'frames')
    labels_path = os.path.join(path, 'labels')
    pickles_path = os.path.join(root_path, 'Pickles')

    data_list = list(paths.list_images(data_path))
    image = cv2.imread(data_list[0])
    width = image.shape[1]
    height = image.shape[0]
    scale_factor = 1 / 8
    s_width = int(width * scale_factor)
    s_height = int(s_width / 2)

    print(s_width, s_height)
    x = make_input('Delete previous data?')

    for folder_path in frames_path, labels_path:
        if os.path.exists(folder_path) and x == 'y':
        # if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    video_list, dst = params()

    data = []
    warp_data = []
    labels = []
    img_labels = []
    warp_labels = []
    previous_frame = []

    i = 0
    j = 0
    for video in video_list:
        name = video['name']
        thresh = video['thresh']
        limit = video['limit']
        src = video['src']

        M = cv2.getPerspectiveTransform(src, dst)
        M_inv = cv2.getPerspectiveTransform(dst, src)

        M_path = os.path.join(pickles_path, f'M_{name}.npy')
        M_inv_path = os.path.join(pickles_path, f'M_inv_{name}.npy')

        np.save(M_path, M)
        np.save(M_inv_path, M_inv)

        for path in data_list[i: i+limit]:
            save_frame = frames_path + fr'\{os.path.basename(path)}'
            save_label = labels_path + fr'\{os.path.basename(path)}'

            frame_exists = os.path.exists(save_frame)
            label_exists = os.path.exists(save_frame)

            image = cv2.imread(path)
            frame = np.copy(image)
            warp, img = prepare(image, src, dst)

            leftx0, lefty0, rightx0, righty0, out_img = find_lanes(img)

            previous_frame = []
            previous_frame.append([leftx0, lefty0, rightx0, righty0])

            left_curve0, right_curve0 = fit_poly(leftx0, lefty0, rightx0, righty0)

            if scale_factor == 1:
                leftx, lefty, rightx, righty = leftx0, lefty0, rightx0, righty0
            else:
                leftx, lefty, rightx, righty = scale_and_perspective(image, left_curve0, right_curve0,
                                                                     src, dst, scale_factor, perspective=False)
            t_leftx, t_lefty, t_rightx, t_righty = scale_and_perspective(image, left_curve0, right_curve0,
                                                                         src, dst, scale_factor, perspective=True)

            left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
            t_left_curve, t_right_curve = fit_poly(t_leftx, t_lefty, t_rightx, t_righty)

            curves = np.concatenate((left_curve, right_curve))
            t_curves = np.concatenate((t_left_curve, t_right_curve))

            labels.append(t_curves)
            warp_labels.append(curves)

            poly, frame = visualise_perspective(frame, left_curve0, right_curve0, src, dst, scale_factor)

            poly = poly / 255
            poly = poly.astype('uint8')
            # import PIL
            # from PIL import ImageOps
            # from keras.preprocessing.image import array_to_img, img_to_array
            # test = np.expand_dims(poly, 2)
            # test = PIL.ImageOps.autocontrast(array_to_img(test))
            # test = img_to_array(test)
            # im_show(test)

            image = cv2.resize(image, (s_width, s_height)) / 255
            image = image.astype('float32')

            warp = cv2.resize(warp, (s_width, s_height)) / 255
            warp = warp.astype('float32')

            data.append(image / 255)
            warp_data.append(warp / 255)
            img_labels.append(poly / 255)

            start = min(min(t_lefty), min(t_righty))
            transformation = visualise(np.copy(warp), left_curve, right_curve, show_lines=True)

            if not frame_exists and not label_exists:
                cv2.imwrite(save_frame, frame)
                cv2.imwrite(save_label, poly)
                print(name, j, path, 'saving frame and label')

            elif not frame_exists and label_exists:
                cv2.imwrite(save_label, poly)
                print(name, j, path, 'saving frame')

            elif frame_exists and not label_exists:
                cv2.imwrite(save_label, poly)
                print(name, j, path, 'saving label')

            else:
                print(name, j, path, 'already processed')

            j += 1

        i += limit
    print('end')

    pickle.dump(labels, open(f'Pickles/{s_width}x{s_height}_labels.p', 'wb'))
    pickle.dump(warp_labels, open(f'Pickles/{s_width}x{s_height}_warp_labels.p', 'wb'))
    pickle.dump(data, open(f'Pickles/{s_width}x{s_height}_data.p', 'wb'))
    pickle.dump(warp_data, open(f'Pickles/{s_width}x{s_height}_warp_data.p', 'wb'))
    pickle.dump(img_labels, open(f'Pickles/{s_width}x{s_height}_img_labels.p', 'wb'))


# path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

# y = make_input('Detect lines?')
# if y=='y':
detect_lines(path)