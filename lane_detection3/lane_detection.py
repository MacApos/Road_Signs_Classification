import os
import cv2
import glob
import time
import pickle
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

start = time.time()

def warp_perspective(image, from_, to):
    M = cv2.getPerspectiveTransform(from_, to)
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warp, M


def threshold(image, T):
    _, image = cv2.threshold(image, T, 250, cv2.THRESH_BINARY)
    return image


def gray_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def brighten(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v>255] = 255

    out_hsv = cv2.merge((h, s, v))
    bright = cv2.cvtColor(out_hsv, cv2.COLOR_HSV2BGR)

    return bright

def color_threshold(image, s_thresh, v_thresh):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    hlv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hlv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1

    return output


def abs_sobel(image, orientation, sobel_thresh):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]

    if orientation == 'x':
        abs_sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    if orientation == 'y':
        abs_sobel = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)

    output = np.zeros_like(abs_sobel)
    output[(abs_sobel >= sobel_thresh[0]) & (abs_sobel <= sobel_thresh[1])] = 1

    return output

def sobel_c_thresh(image):
    output = np.zeros_like(image[:, :, 1])
    sobel_x = abs_sobel(image, "x", (25, 255))
    sobel_y = abs_sobel(image, "y", (25, 255))
    c_binary = color_threshold(image, (25, 255), (25, 255))
    output[(sobel_x==1)&(sobel_y==1) | (c_binary==1)]=250

    return output


def draw_lines(image, arr, point_color=(255, 0, 0), line_color=(0, 255, 0)):
    arr = arr.astype(int)
    copy = np.copy(image)
    for i in range(arr.shape[0]):
        x, y = arr[i][0], arr[i][1]
        x_0, y_0 = arr[i - 1][0], arr[i - 1][1]
        cv2.circle(copy, (x, y), radius=1, color=point_color, thickness=10)
        cv2.line(copy, (x, y), (x_0, y_0), color=line_color, thickness=4)
    return copy


def find_contours(image, display=False):
    contours, hierarchy = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    edges = np.zeros((image.shape[0], image.shape[1]))
    cv2.drawContours(edges, contours, -1, (255), 1)
    if display:
        cv2.imshow('contours', edges)
        cv2.waitKey(0)
    return contours, edges


def to_csv(arr, name):
    df = pd.DataFrame(arr)
    path = os.path.join('../lane_detection2/Arrays', name)
    df.to_csv(path, sep='\t', index=False, header=False)


def to_jpg(name, image):
    path = 'Pictures/'+name+'.jpg'
    cv2.imwrite(path, image)


def prepare(image, thresh_flag=True, sobel_flag=False):
    global contours
    box = draw_lines(image, src)
    box = draw_lines(box, dst, line_color=(0, 0, 255))
    warp, _ = warp_perspective(image, src, dst)
    gray = gray_img(warp)

    max_val = np.mean(np.amax(gray, axis=1)).astype(int)
    # max_val = int((1 - (0.5 * math.log(max_val) - 2.13)) * max_val)
    if max_val > (255*0.75):
        max_val = int(max_val*0.75)

    thresh = threshold(gray, max_val)

    ''' [...] Any edges with intensity gradient more than maxVal are sure to be edges and those below minVal are sure
    to be non-edges, so discarded '''
    canny = cv2.Canny(thresh, 0, 75)
    ''' [...] threshold of the minimum number of intersections needed to detect a line. '''
    lines = cv2.HoughLinesP(canny, 2, np.pi/180, 50, np.array([]), minLineLength=15, maxLineGap=5)

    left_lane = []
    right_lane = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if -0.1 > slope or slope > 0.1:
                if x1 <= width//2 or x2 <= width//2:
                    left_lane.append((slope, intercept))
                    cv2.line(warp, (x1, y1), (x2, y2), (0, 255, 0), 4)
                else:
                    right_lane.append((slope, intercept))
                    cv2.line(warp, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # to_jpg('box', box)
    # to_jpg('warp', warp)
    # to_jpg('gray', gray)
    # to_jpg('threshold', thresh)
    # to_jpg('contours', img)
    # to_jpg('Houg', warp)

    return thresh, left_lane, right_lane


def find_lanes(image):
    global left_mean, right_mean

    lane_lists = []
    if left_lane:
        left_mean = np.mean(left_lane, axis=0)
        left_slope = left_mean[0]
        left_intercept = left_mean[1]
        lane_lists.append(left_mean)

    if right_lane:
        right_mean = np.mean(right_lane, axis=0)
        right_slope = right_mean[0]
        right_intercept = right_mean[1]
        lane_lists.append(right_mean)

    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    out_img = np.dstack((image, image, image))*225

    midpoint = int(histogram.shape[0]//2)
    left = np.argmax(histogram[:midpoint])
    right = midpoint + np.argmax(histogram[midpoint:])

    if left > midpoint or (midpoint - left) < 100:
        left = 0 + margin
    if right < midpoint or (right - midpoint) < 100:
        right = width - margin - 1

    left_current = left
    right_current = right

    left_idx = []
    right_idx = []

    nonzero = np.nonzero(image)
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    try:
        for list in lane_lists:
            slope = list[0]
            intercept = list[1]
            if slope:
                y1 = 0
                x1 = int((y1 - intercept) / slope)
                y2 = height
                x2 = int((y2 - intercept) / slope)
                cv2.line(out_img, (x1, y1), (x2, y2), (255, 0, 0), 4)
    except ValueError:
        pass

    # to_jpg('line', out_img)

    for i in range(number):
        low = image.shape[0] - win_height*(i+1)
        high = image.shape[0] - win_height*i
        left_left = left_current - margin
        left_right = left_current + margin
        right_left = right_current - margin
        right_right = right_current + margin

        cv2.rectangle(out_img, (left_left, low), (left_right, high), (0, 255, 0), 4)
        cv2.rectangle(out_img, (right_left, low), (right_right, high), (0, 255, 0), 4)

        left_nonzero = ((nonzeroy >= low) & (nonzeroy <= high) &
                        (nonzerox >= left_left) & (nonzerox <= left_right)).nonzero()[0]

        right_nonzero = ((nonzeroy >= low) & (nonzeroy <= high) &
                         (nonzerox >= right_left) & (nonzerox <= right_right)).nonzero()[0]

        left_idx.append(left_nonzero)
        right_idx.append(right_nonzero)

        if len(left_nonzero) > minpix:
            left_current = int(np.mean(nonzerox[left_nonzero]))
        else:
            try:
                left_current = int((low - left_intercept) / left_slope)
                # print('follow left line')
            except NameError:
                pass

        if len(right_nonzero) > minpix:
            right_current = int(np.mean(nonzerox[right_nonzero]))
        else:
            try:
                # print('follow right line')
                right_current = int((low - right_intercept) / right_slope)
            except NameError:
                pass

    # to_jpg('rectangles', out_img)

    try:
        left_idx = np.concatenate(left_idx)
        right_idx = np.concatenate(right_idx)
    except AttributeError:
        pass

    leftx1 = nonzerox[left_idx]
    lefty1 = nonzeroy[left_idx]
    rightx1 = nonzerox[right_idx]
    righty1 = nonzeroy[right_idx]

    if len(leftx1) == 0:
        # leftx1 = rightx1 - width // 2
        leftx1 = width - rightx1
        lefty1 = righty1

    if len(rightx1) == 0:
        # rightx1 = leftx1 + width // 2
        rightx1 = width - leftx1
        righty1 = lefty1

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

    if len(leftx)==0 or len(rightx)==0:
        leftx, lefty, rightx, righty = leftx1, lefty1, rightx1, righty1

    return leftx, lefty, rightx, righty, out_img


def find_lanes_perspective():
    global M_inv
    zeros = np.zeros_like(frame)
    zeros[lefty, leftx] = 255
    zeros[righty, rightx] = 255
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
    frame = cv2.addWeighted(frame, 1, poly, 0.6, 0)

    return poly, frame


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# thresh, th_left_lane, th_right_lane = prepare(image)
# v1 = [thresh, th_left_lane, th_right_lane]
#
# sobel, s_left_lane, s_right_lane = prepare(image, thresh_flag=False, sobel_flag=True)
# v2 = [sobel, s_left_lane, s_right_lane]
#
# fig, axs = plt.subplots(2, 2, figsize=(20,20))
# [ax.set_axis_off() for ax in axs.ravel()]

# title = 'thresh'
# for idx, val in enumerate([v1, v2]):
#     img = val[0]
#     left_lane = val[1]
#     right_lane = val[2]

data_path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
path = os.path.join(data_path, 'train_set')
list_dir = os.listdir(path)

random = random.randint(0, len(list_dir)-1)
# random = 359
print(random)

image = cv2.imread(os.path.join(path, fr'{random}.jpg'))

image = cv2.resize(image, (1280, 720))
frame = image
height = image.shape[0]
width = image.shape[1]


# src = np.float32([[0, 720],
#                   [450, 300],
#                   [850, 300],
#                   [1280, 720]])

# dst = np.float32([src[0],
#                   [src[0][0], 0],
#                   [src[-1][0], 0],
#                   src[-1]])

src = np.float32([[290,675],
                  [570,525],
                  [710,525],
                  [990,675]])

dst = pts2 = np.float32([[0,height],
                         [0,0],
                         [width,0],
                         [width,height]])

# file = open('Pickles/src.p', 'rb')
# src = pickle.load(file)
# file.close()
number = 9
minpix = 50
margin = 100
win_height = int(image.shape[0] // number)

train_line = []
train_t_line = []
train_poly = []

test_line = []
test_t_line = []
test_poly = []

labels = ['train_set', 'test_set']

for label in labels:
    label_path = data_path + fr'\{label}'

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        os.mkdir(data_path)
    else:
        os.mkdir(data_path)

    filepath = glob.glob(label_path + '\\*.jpg')
    print(len(filepath))

    if label == 'train_set':
        save_path = data_path + fr'\train_labels'
        save_line = train_line
        save_t_line = train_t_line
        save_poly = train_poly

    elif label == 'test_set':
        save_path = data_path + fr'\test_labels'
        save_line = test_line
        save_t_line = test_t_line
        save_poly = test_poly

    print(save_path+fr'\{1}.jpg')

    for idx, val in enumerate(filepath):
        print(val)
        image = cv2.imread(os.path.join(label_path, f'{idx}.jpg'))
        frame = image

        img, left_lane, right_lane = prepare(image, thresh_flag=True, sobel_flag=False)
        nonzerox = len(img.nonzero()[1])

        if nonzerox < 6500:
            image = brighten(image, 25)
            img, left_lane, right_lane = prepare(image, thresh_flag=True, sobel_flag=False)
        else:
            pass

        copy = np.copy(img)
        leftx, lefty, rightx, righty, out_img = find_lanes(copy)

        t_leftx, t_lefty, t_rightx, t_righty, _ = find_lanes_perspective()

        left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
        t_left_curve, t_right_curve = fit_poly(t_leftx, t_lefty, t_rightx, t_righty)

        curves = np.concatenate((left_curve, right_curve))
        t_curves = np.concatenate((t_left_curve, t_right_curve))

        # Visualisation
        y = np.linspace(0, height-1, 15).astype(int).reshape((-1,1))
        out_img, fit_leftx, fit_rightx, points = visualise(out_img, y, left_curve, right_curve, False)

        down = min(min(t_lefty), min(t_righty))
        t_y = np.linspace(down, 720, 15).astype(int).reshape((-1,1))
        t_out_img, fit_t_leftx, fit_t_rightx, _ = visualise(np.copy(image), t_y, t_left_curve, t_right_curve, False)

        poly, frame = visualise_perspective(frame)
        poly = poly[:,:,1]

        cv2.imwrite(save_path+fr'\{idx}.jpg', frame)
        save_line.append(curves)
        save_t_line.append(t_curves)
        save_poly.append(poly)

print()


# axs[0][idx].imshow(rgb(frame_processed))
# axs[0][idx].set_title(title)
# axs[1][idx].imshow(rgb(out_img))
#
# title = 'sobel'

# plt.show()
# to_jpg('original', image)
# to_jpg('perspective', t_out_img)
# to_jpg('curves', out_img)
# to_jpg('frame', frame)

# cv2.imshow('frame', frame)
# cv2.waitKey(0)
# cv2.imshow('out_img', out_img)
# cv2.waitKey(0)