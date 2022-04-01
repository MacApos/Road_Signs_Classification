import os
import cv2
import random
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
list_dir = os.listdir(path)
random = random.randint(0, len(list_dir)-1)
# random = 754
print(random)
image = cv2.imread(os.path.join(path, f'{random}.jpg'))

image = cv2.resize(image, (1280, 720))
image = cv2.flip(image, 1)
frame = image

height = image.shape[0]
width = image.shape[1]

number = 9
minpix = 50
margin = 100
win_height = int(image.shape[0]//number)


src = np.float32([[0, 720],
                  [450, 300],
                  [850, 300],
                  [1280, 720]])

dst = np.float32([src[0],
                  [src[0][0], 0],
                  [src[-1][0], 0],
                  src[-1]])


def warp_perspective(image, from_=src, to=dst):
    M = cv2.getPerspectiveTransform(from_, to)
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warp, M


def threshold(image, T):
    _, image = cv2.threshold(image, T, 250, cv2.THRESH_BINARY)
    return image


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


def find_contours(image, display=False):
    contours = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = list(imutils.grab_contours(contours))
    edges = np.zeros((image.shape[0], image.shape[1]))
    for contour in contours:
        cv2.drawContours(image=edges, contours=[contour], contourIdx=-1, color=(255), thickness=1)
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

to_jpg('original', image)

def prepare(image):
    global contours

    box = draw_lines(image, src)
    to_jpg('box', box)
    box = draw_lines(box, dst, line_color=(0, 0, 255))
    warp, _ = warp_perspective(image)
    to_jpg('warp', warp)
    gray = gray_img(warp)
    to_jpg('gray', gray)
    max_val = np.mean(np.amax(gray, axis=1)).astype(int)

    if max_val > (255*0.75):
        max_val = int(max_val*0.75)

    image = threshold(gray, max_val)
    to_jpg('threshold', image)
    contours, img = find_contours(image, display=False)
    to_jpg('contours', img)
    gray_rgb = np.dstack((image, image, image)) * 225
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minpix:
            cv2.drawContours(image=gray_rgb, contours=[contour], contourIdx=-1, color=(0, 0, 0), thickness=-1)

    try:
        left_max = np.nonzero(gray_rgb[:, :width // 2])[0].max()
    except ValueError:
        left_max = height - 1

    try:
        right_max = np.nonzero(gray_rgb[:, width // 2:])[0].max()
    except ValueError:
        right_max = height - 1

    down = min((left_max, right_max))
    if down < int(0.3*height):
        down = int(0.3 * height)

    blur = cv2.GaussianBlur(gray[down-int(0.3*height): down, :], (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(canny, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=5)

    left_lane = []
    right_lane = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            y1 = y1 + down-int(0.3*height)
            y2 = y2 + down-int(0.3*height)
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

    to_jpg('Houg', warp)

    return image, left_lane, right_lane


def find_lanes(image, drop_out=True):
    global left_mean, right_mean

    lane_lists = []
    if left_lane or right_lane:
        if drop_out:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < minpix:
                    cv2.drawContours(image=image, contours=[contour], contourIdx=-1, color=(0), thickness=-1)
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

    to_jpg('line', out_img)

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
                print('follow left line')
            except NameError:
                pass

        if len(right_nonzero) > minpix:
            right_current = int(np.mean(nonzerox[right_nonzero]))
        else:
            try:
                right_current = int((low - right_intercept) / right_slope)
                print('follow right line')
            except NameError:
                pass

    to_jpg('rectangles', out_img)

    try:
        left_idx = np.concatenate(left_idx)
        right_idx = np.concatenate(right_idx)
    except AttributeError:
        pass

    leftx0 = nonzerox[left_idx]
    lefty0 = nonzeroy[left_idx]
    rightx0 = nonzerox[right_idx]
    righty0 = nonzeroy[right_idx]

    if len(leftx0) == 0:
        leftx0 = rightx0 - width // 2
        lefty0 = righty0

    if len(rightx0) == 0:
        rightx0 = leftx0 + width // 2
        righty0 = lefty0

    left_curve0 = np.polyfit(lefty0, leftx0, 2)
    right_curve0 = np.polyfit(righty0, rightx0, 2)

    left_nonzero1 = (
                (nonzerox > (left_curve0[0] * (nonzeroy ** 2) + left_curve0[1] * nonzeroy + left_curve0[2] - margin)) &
                (nonzerox < (left_curve0[0] * (nonzeroy ** 2) + left_curve0[1] * nonzeroy + left_curve0[2] + margin)))

    right_nonzero1 = (
            (nonzerox > (right_curve0[0] * (nonzeroy ** 2) + right_curve0[1] * nonzeroy + right_curve0[2] - margin)) &
            (nonzerox < (right_curve0[0] * (nonzeroy ** 2) + right_curve0[1] * nonzeroy + right_curve0[2] + margin)))

    leftx = nonzerox[left_nonzero1]
    lefty = nonzeroy[left_nonzero1]
    rightx = nonzerox[right_nonzero1]
    righty = nonzeroy[right_nonzero1]

    if len(leftx)==0 or len(rightx)==0:
        leftx, lefty, rightx, righty = leftx0, lefty0, rightx0, righty0

    return leftx, lefty, rightx, righty, out_img


def find_lanes_perspective():
    global M_inv
    zeros = np.zeros_like(frame)
    zeros[lefty, leftx] = 255
    zeros[righty, rightx] = 255
    _, M_inv = warp_perspective(frame, from_=dst, to=src)
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
    left_x = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
    right_x = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]

    empty = []
    flipud = False

    for arr in left_x, right_x:
        arr = arr.astype(int)
        con = np.concatenate((arr, y), axis=1)
        cv2.polylines(image, [con], isClosed=False, color=(0, 0, 255), thickness=4)
        if flipud:
            con = np.flipud(con)
        flipud = True
        empty.append(con)

    points = np.vstack((empty[0], empty[1]))

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(leftx, -lefty, c='g')
        ax.scatter(rightx, -righty, c='r')
        ax.plot(left_x, -y, c='b')
        ax.plot(right_x, -y, c='b')
        plt.show()

    return image, left_x, right_x, points


def visualise_perspective(frame):
    poly = np.dstack((image, image, image)) * 255
    poly = cv2.fillPoly(poly, [points], (0, 255, 0))

    poly = cv2.warpPerspective(poly, M_inv, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    frame = cv2.addWeighted(frame, 1, poly, 0.6, 0)

    return frame


image, left_lane, right_lane = prepare(image)

copy = np.copy(image)
leftx1, lefty1, rightx1, righty1, out_img1 = find_lanes(copy, True)

if len(leftx1)<400 or len(rightx1)<400:
    leftx, lefty, rightx, righty, out_img = find_lanes(image, False)
else:
    leftx, lefty, rightx, righty, out_img = leftx1, lefty1, rightx1, righty1, out_img1

t_leftx, t_lefty, t_rightx, t_righty, t_out_img = find_lanes_perspective()

# to_jpg('perspective', t_out_img)

left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
t_left_curve, t_right_curve = fit_poly(t_leftx, t_lefty, t_rightx, t_righty)

# Visualisation
y = np.linspace(0, height-1, height).astype(int).reshape((-1,1))
out_img, left_x, right_x, points = visualise(out_img, y, left_curve, right_curve, True)

down = min(min(t_lefty), min(t_righty))
up = max(max(t_lefty), max(t_righty))
t_y = np.linspace(down, up, (up-down)).astype(int).reshape((-1,1))
t_out_img, t_left_x, t_right_x, _ = visualise(t_out_img, t_y, t_left_curve, t_right_curve, False)

to_jpg('curves', out_img)

frame = visualise_perspective(frame)

to_jpg('frame', frame)

# cv2.imshow('image', image)
# cv2.waitKey(0)
cv2.imshow('t_out_img', t_out_img)
cv2.waitKey(0)
cv2.imshow('out_img', out_img)
cv2.waitKey(0)
cv2.imshow('frame', frame)
cv2.waitKey(0)