import os
import cv2
import time
import pickle
import random
import shutil
import numpy as np
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt

start = time.time()


def im_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)


def warp_perspective(image, from_, to):
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


def to_csv(name, arr):
    df = pd.DataFrame(arr)
    path = os.path.join('../lane_detection2/Arrays', name)
    df.to_csv(path, sep='\t', index=False, header=False)


def to_jpg(name, image):
    path = 'Pictures/' + name + '.jpg'
    cv2.imwrite(path, image)


def prepare(image, thresh):
    box = draw_lines(image, src)
    box = draw_lines(box, dst, line_color=(0, 0, 255))
    warp, _ = warp_perspective(image, src, dst)
    gray = gray_img(warp)
    max_val = max(np.amax(gray, axis=1)).astype(int)
    thresh = color_mask(warp, (max_val * 0.65, max_val))

    return thresh


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
    global out_img
    global low, high
    global nonzerox, nonzeroy
    global left_intercept, left_slope, right_intercept, right_slope

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


def find_lanes_perspective():
    global M_inv
    zeros = np.zeros_like(frame)
    zeros[lefty - 1, leftx - 1] = 255
    zeros[righty - 1, rightx - 1] = 255
    _, M_inv = warp_perspective(frame, dst, src)
    t_out_img = cv2.warpPerspective(zeros, M_inv, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

    t_leftx = t_out_img[:, :width // 2].nonzero()[1]
    t_lefty = t_out_img[:, :width // 2].nonzero()[0]
    t_rightx = t_out_img[:, width // 2:].nonzero()[1] + width // 2
    t_righty = t_out_img[:, width // 2:].nonzero()[0]

    if len(t_leftx) == 0:
        t_leftx = -t_rightx + width
        t_lefty = t_righty

    if len(t_rightx) == 0:
        t_rightx = -t_leftx + width
        t_righty = t_lefty

    return t_leftx, t_lefty, t_rightx, t_righty, t_out_img


def fit_poly(leftx, lefty, rightx, righty):
    left_curve = np.polyfit(lefty, leftx, 2)
    right_curve = np.polyfit(righty, rightx, 2)

    return left_curve, right_curve


def visualise(image, y, left_curve, right_curve, plot):
    fit_leftx = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
    fit_rightx = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]

    empty = []
    flipud = False

    color = (0, 0, 255)
    for arr in fit_leftx, fit_rightx:
        arr = arr.astype(int)
        con = np.concatenate((arr, y), axis=1)
        # cv2.polylines(image, [con], isClosed=False, color=(0, 0, 255), thickness=4)

        for val in con:
            cv2.circle(image, tuple(val), 5, color, -1)

        if flipud:
            con = np.flipud(con)
        flipud = True
        empty.append(con)

        color = (255, 0, 0)

    points = np.vstack((empty[0], empty[1]))

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(leftx, -lefty, c='g')
        ax.scatter(rightx, -righty, c='r')
        ax.plot(fit_leftx, -y, c='b')
        ax.plot(fit_rightx, -y, c='b')
        plt.show()

    return image, fit_leftx, fit_rightx, points


def visualise_perspective(frame):
    poly = np.dstack((img, img, img)) * 255
    cv2.fillPoly(poly, [points], (0, 255, 0))
    poly = cv2.warpPerspective(poly, M_inv, (poly.shape[1], poly.shape[0]), flags=cv2.INTER_LINEAR)
    out_frame = cv2.addWeighted(frame, 1, poly, 0.6, 0)
    poly = poly[:, :, 1]

    return poly, out_frame


def sort_path(path):
    sorted_path = []
    for file in os.listdir(path):
        number = int(''.join(n for n in file if n.isdigit()))
        sorted_path.append(number)

    sorted_path = sorted(sorted_path)
    return [path + fr'\{str(f)}.jpg' for f in sorted_path]


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
# data_path = os.path.join(path, 'data')
# labels_path = os.path.join(path, 'labels')

raw = ['data', 'frames', 'labels']
augmented = ['augmented_data', 'augmented_frames', 'augmented_labels']

for folder in [raw]:
    # path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
    # path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
    path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
    data_path = os.path.join(path, folder[0])
    frames_path = os.path.join(path, folder[1])
    labels_path = os.path.join(path, folder[2])

    for folder_path in frames_path, labels_path:
        # shutil.rmtree(folder_path)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    data_list = list(paths.list_images(data_path))

    image = cv2.imread(data_list[0])
    height = image.shape[0]
    width = image.shape[1]

    template = [[280, 650], [570, 525]]

    src = np.float32([template[0],
                      template[1],
                      [width - template[1][0], template[1][1]],
                      [width - template[0][0], template[0][1]]])

    dst = np.float32([[0, height],
                      [0, 0],
                      [width, 0],
                      [width, height]])

    video1 = {'name': 'video1',
              'template': [[290, 390], [550, 265]],
              'thresh': 0.65,
              'limit': 2548}  #

    video2 = {'name': 'video2',
              'template': [[285, 390], [550, 265]],
              'thresh': 0.55,
              'limit': 4122}  #

    video3 = {'name': 'video3',
              'template': [[280, 400], [570, 230]],
              'thresh': 0.85,
              'limit': 2833}

    video4 = {'name': 'video4',
              'template': [[270, 400], [550, 245]],
              'thresh': 0.9,
              'limit': 1840}

    video_list = [video1]  # video2

    number = 35
    minpix = 50
    margin = 100
    win_height = int(height // number)

    label_list = []
    previous_frame = []

    i = 0
    j = 0
    for video in video_list:
        values = list(video.values())
        name = values[0]
        template = values[1]
        thresh = values[2]
        limit = values[3]

        src = np.float32([template[0],
                          template[1],
                          [width - template[1][0], template[1][1]],
                          [width - template[0][0], template[0][1]]])
        # i: i + limit
        for path in data_list[:10]:
            save_frame = frames_path + fr'\{os.path.basename(path)}'
            save_label = labels_path + fr'\{os.path.basename(path)}'
            # if os.path.exists(save_frame) and os.path.exists(save_label):
            #     print(i, j, {path}, 'already processed')
            #     j += 1
            #     continue

            image = cv2.imread(path)
            frame = image

            img = prepare(image, thresh)

            copy = np.copy(img)
            leftx, lefty, rightx, righty, out_img = find_lanes(copy)
            t_leftx, t_lefty, t_rightx, t_righty, _ = find_lanes_perspective()

            previous_frame = []
            previous_frame.append([leftx, lefty, rightx, righty])

            left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
            t_left_curve, t_right_curve = fit_poly(t_leftx, t_lefty, t_rightx, t_righty)

            curves = np.concatenate((left_curve, right_curve))
            t_curves = np.concatenate((t_left_curve, t_right_curve))

            # Visualisation
            y = np.linspace(0, height - 1, 15).astype(int).reshape((-1, 1))
            out_img, fit_leftx, fit_rightx, points = visualise(out_img, y, left_curve, right_curve, False)
            # down = min(min(t_lefty), min(t_righty))
            # t_y = np.linspace(down, 720, 15).astype(int).reshape((-1,1))
            # t_out_img, fit_t_leftx, fit_t_rightx, _ = visualise(np.copy(image), t_y, t_left_curve, t_right_curve, False)

            poly, frame = visualise_perspective(frame)

            im_show('out_img', frame)

            # if not os.path.exists(save_frame):
            #     cv2.imwrite(save_frame, frame)
            #
            # if not os.path.exists(save_label):
            #     cv2.imwrite(save_label, poly)

            # print(i, j, 'save', path)
            label_list.append(curves)

            j += 1

        i += limit
    print('next')

    pickle.dump(label_list, open(f'Pickles/{folder[2]}.p', "wb"))
