import os
import cv2
import pickle
import shutil
import numpy as np
import pandas as pd
from imutils import paths
import matplotlib.pyplot as plt
# from keras import img_to_array
from keras.preprocessing.image import img_to_array


def im_show(image, name='Image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)


def to_csv(name, arrqy):
    array_list = []
    for arr in arrqy:
        if isinstance(arr, np.ndarray):
            arr = arr[0]
        arr = str(arr).replace('.', ',')
        array_list.append(arr)

    df = pd.DataFrame(array_list)
    # path = os.path.join(r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3\Arrays', name)
    path = os.path.join(r'/lane_detection3/Arrays', name)
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


def prepare(image, src, dst, width, height):
    box = draw_lines(image, src)
    box = draw_lines(box, dst, line_color=(0, 0, 255))
    warp, _ = warp_perspective(image, src, dst, width, height)
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
    global margin, minpix
    global out_img
    global low, high
    global nonzerox, nonzeroy
    global left_intercept, left_slope, right_intercept, right_slope
    global lefty, leftx, righty, rightx

    number = 35
    minpix = int((width * height) / 11776)
    margin = int(width / 12.8)

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

    for idx, arr in enumerate([fit_left, fit_right]):
        arr = arr.astype(int)
        con = np.concatenate((arr, y), axis=1)

        if flipud:
            con = np.flipud(con)

        flipud = True
        empty.append(con)

    points = np.vstack((empty[0], empty[1]))
    points = np.split(points, 2, 0)

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



def visualise_perspective(image, left_curve, right_curve, src, dst, show_lines=False, show_points=False):
    poly = np.zeros_like(image) * 255
    lines_img = np.copy(poly)
    points_img = np.copy(poly)
    width = poly.shape[1]
    height = poly.shape[0]

    M_inv = cv2.getPerspectiveTransform(dst, src)
    points_arr = generate_points(image, left_curve, right_curve)

    colors = [(255, 0, 0), (0, 0, 255)]
    for idx, arr in enumerate(points_arr):
        cv2.polylines(lines_img, [arr], isClosed=False, color=colors[idx], thickness=25)

        zeros = np.zeros_like(poly)
        for point in arr:
            cv2.circle(points_img, tuple(point), 15, colors[idx], -1)
            cv2.circle(zeros, tuple(point), 1, 1, -1)

        zeros = cv2.warpPerspective(zeros, M_inv, (width, height), flags=cv2.INTER_LINEAR)
        nonzero = np.nonzero(zeros)

        nonzerox = np.array(nonzero[1]).reshape((-1, 1))
        nonzeroy = np.array(nonzero[0]).reshape((-1, 1))

        con = np.concatenate((nonzerox, nonzeroy), axis=1)

        # if show_lines:
        #     cv2.polylines(frame, [con], isClosed=False, color=colors[idx], thickness=5)
        #
        # if show_points:
        #     for val in con:
        #         cv2.circle(frame, tuple(val), 5, colors[idx], -1)

    if show_lines:
        lines_img = cv2.warpPerspective(lines_img, M_inv, (width, height), flags=cv2.INTER_LINEAR)
        image = cv2.addWeighted(image, 1, lines_img, 1, 0)

    if show_points:
        points_img = cv2.warpPerspective(points_img, M_inv, (width, height), flags=cv2.INTER_LINEAR)
        image = cv2.addWeighted(image, 1, points_img, 1, 0)

    points = np.vstack((points_arr[0], points_arr[1]))
    cv2.fillPoly(poly, [points], (0, 255, 0))
    poly = cv2.warpPerspective(poly, M_inv, (poly.shape[1], poly.shape[0]), flags=cv2.INTER_LINEAR)
    out_frame = cv2.addWeighted(image, 1, poly, 0.5, 0)
    poly = poly[:, :, 1]

    return poly, out_frame


def find_lanes_perspective(image):
    t_left = np.zeros_like(image)
    t_right = np.zeros_like(image)
    t_left[lefty - 1, leftx - 1] = 255
    t_right[righty - 1, rightx - 1] = 255

    t_left = cv2.warpPerspective(t_left, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    t_right = cv2.warpPerspective(t_right, M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    t_out_img = cv2.addWeighted(t_left, 1, t_right, 1, 0)

    t_leftx = t_left.nonzero()[1]
    t_lefty = t_left.nonzero()[0]
    t_rightx = t_right.nonzero()[1]
    t_righty = t_right.nonzero()[0]

    if len(t_leftx) == 0:
        t_leftx = -t_rightx + width
        t_lefty = t_righty

    if len(t_rightx) == 0:
        t_rightx = width-t_leftx
        t_righty = t_lefty

    return t_leftx, t_lefty, t_rightx, t_righty, t_out_img


def params(width, height):
    scale = width / 1280

    video1 = {'name': 'video1',
              'src': np.float32([[290, 410], [550, 285]]) * scale,
              'thresh': 0.65,
              'limit': 2548}

    video2 = {'name': 'video2',
              'src': np.float32([[285, 410], [550, 285]]) * scale,  # [[:, 390+20], [:, 265+20]]
              'thresh': 0.55,
              'limit': 4122}

    video3 = {'name': 'video3',
              'src': np.float32([[280, 420], [570, 250]]) * scale,
              'thresh': 0.85,
              'limit': 2833}

    video4 = {'name': 'video4',
              'src': np.float32([[270, 420], [550, 265]]) * scale,
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

def detect_lines(path, folder):
    global width, height
    global previous_frame
    global M_inv

    data_path = os.path.join(path, folder[0])
    frames_path = os.path.join(path, folder[1])
    labels_path = os.path.join(path, folder[2])
    data_npy = r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\small_data.npy'
    # data_npy = r'F:\krzysztof\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\data.npy'
    data_list = list(paths.list_images(data_path))

    x = make_input('Delete previous data?')

    for folder_path in frames_path, labels_path:
        if os.path.exists(folder_path) and x == 'y':
            shutil.rmtree(folder_path)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    image = cv2.imread(data_list[0])
    scale_factor = 1/4
    image = cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)))
    width = image.shape[1]
    height = image.shape[0]

    video_list, dst = params(width, height)

    data = []
    labels = []
    previous_frame = []

    i = 0
    j = 0
    for video in video_list:
        name = video['name']
        thresh = video['thresh']
        limit = video['limit']
        src = video['src']

        for path in data_list[i: i+100]:
            save_frame = frames_path + fr'\{os.path.basename(path)}'
            save_label = labels_path + fr'\{os.path.basename(path)}'

            frame_exists = os.path.exists(save_frame)
            label_exists = os.path.exists(save_frame)

            # if frame_exists and label_exists:
            #     print(name, j, path, 'already processed')
            #     j += 1
            #     continue

            image = cv2.imread(path)
            image = cv2.resize(image, (width, height))
            data.append(image)

            frame = np.copy(image)

            img = prepare(image, src, dst, width, height)

            M_inv = cv2.getPerspectiveTransform(dst, src)
            leftx, lefty, rightx, righty, out_img = find_lanes(img)
            t_leftx, t_lefty, t_rightx, t_righty, t_out_img = find_lanes_perspective(image)

            previous_frame = []
            previous_frame.append([leftx, lefty, rightx, righty])

            left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
            t_left_curve, t_right_curve = fit_poly(t_leftx, t_lefty, t_rightx, t_righty)

            curves = np.concatenate((left_curve, right_curve))
            t_curves = np.concatenate((t_left_curve, t_right_curve))
            labels.append(curves)

            start = min(min(t_lefty), min(t_righty))
            # out_img = visualise(out_img, left_curve, right_curve, show_lines=True)
            # t_out_img = visualise(image, t_left_curve, t_right_curve, start, show_lines=True)
            poly, frame = visualise_perspective(frame, left_curve, right_curve, src, dst)
            test(frame, left_curve, right_curve, src, dst)
            # im_show(frame)

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
    print('next')

    pickle.dump(labels, open(f'../Pickles/small_labels.p', "wb"))
    data = np.array(data, dtype='float') / 255.
    np.save(data_npy, data)


# path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
# path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'

raw = ['data', 'frames', 'labels']
augmented = ['augmented_data', 'augmented_frames', 'augmented_labels']

y = make_input('Detect lines?')
if y=='y':
    detect_lines(path, raw)