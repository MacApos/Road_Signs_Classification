import os
import cv2
import math
import random
import numpy as np
import pandas as pd
import imutils
import matplotlib.pyplot as plt


path = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
# path = r'F:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
list_dir = os.listdir(path)
# random = random.randint(0, len(list_dir)-1)
random = 2950
print(random)
image = cv2.imread(os.path.join(path, f'{random}.jpg'))
cv2.imshow('image', image)
cv2.waitKey(0)
# image = cv2.resize(image, (1280, 720))
# image = cv2.resize(image, (640, 480))
image = cv2.flip(image, 1)
frame = image

height = image.shape[0]
width = image.shape[1]

number = 9
minpix = 50
margin = 100
win_height = int(image.shape[0]//number)

# np
# height ↓ row
# width  –→ col

# cv
# width  –→ col
# height ↓ row

#
# src = np.float32([(515, 460),
#                   (150, 660),
#                   (1130, 660),
#                   (765, 460)])
#
# dst = np.float32([(100, 0),
#                   (100, 720),
#                   (1100, 720),
#                   (1100, 0)])
#

# src = np.float32([(175, 325),
#                   (300, 175),
#                   (350, 175),
#                   (500, 325)])

# dst = np.float32([(src[0]),
#                   (src[0][0], src[1][1]),
#                   (src[-1][0], src[2][1]),
#                   (src[-1])])

# width = int(src[3][0]-src[0][0])
# height = int(src[3][1] - src[1][1])


src = np.float32([(0, 720),
                  (400, 300),
                  (850, 300),
                  (1280, 720)])

dst = np.float32([(src[0]),
                  (src[0][0], 0),
                  (src[-1][0], 0),
                  (src[-1])])


def warp_perspective(image, from_=src, to=dst):
    # M → from_=src, to=dst
    # M_inv → from_=dst, to=src
    M = cv2.getPerspectiveTransform(from_, to)
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warp, M

# def warp_perspective_inv(image):
#     M_inv = cv2.getPerspectiveTransform(dst, src)
#     warp_inv = cv2.warpPerspective(image, M_inv, (width, height), flags=cv2.INTER_LINEAR)
#
#     return warp_inv, M_inv


def threshold(image, T):
    _, image = cv2.threshold(image, T, 250, cv2.THRESH_BINARY)
    # (_, image) = cv2.threshold(image, 0, 225, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
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
        cv2.line(copy, (x, y), (x_0, y_0), color=line_color, thickness=5)
        # cv2.putText(copy, f'{x}, {y}', (x - 100, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return copy


def find_contours(image, display=False):
    contours = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = list(imutils.grab_contours(contours))
    edges = np.zeros((image.shape[0], image.shape[1]))
    for contour in contours:
        cv2.drawContours(image=edges, contours=[contour], contourIdx=-1, color=(255),
                         thickness=1)
    if display:
        cv2.imshow('contours', edges)
        cv2.waitKey(0)
    return contours, edges


def to_csv(arr, name):
    df = pd.DataFrame(arr)
    path = os.path.join('../../lane_detection2/Arrays', name)
    df.to_csv(path, sep='\t', index=False, header=False)


def to_jpg(image, name):
    path = os.path.join('../../lane_detection2/test', (name + '.jpg'))
    cv2.imwrite(path, image)


# img = image
def prepare(image):
    global contours
    box = draw_lines(image, src)
    box = draw_lines(box, dst, line_color=(0, 0, 255))
    warp, _ = warp_perspective(image)

    gray = gray_img(warp)
    max_val = np.mean(np.amax(gray, axis=1)).astype(int)

    if max_val > (255*0.75):
        max_val = int(max_val*0.75)

    image = threshold(gray, max_val)

    contours, _ = find_contours(image, display=False)
    gray_rgb = np.dstack((image, image, image)) * 225
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minpix:
            cv2.drawContours(image=gray_rgb, contours=[contour], contourIdx=-1, color=(0, 0, 0), thickness=-1)

    try:
        left_min = np.nonzero(gray_rgb[:, :width // 2])[0].max()
    except ValueError:
        left_min = height - 1

    try:
        right_min = np.nonzero(gray_rgb[:, width // 2:])[0].max()
    except ValueError:
        right_min = height - 1

    down = min((left_min, right_min))
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

                # if slope < -0.1: # /_ <- slope
                #     left_lane.append((slope, intercept))
                #     cv2.line(warp, (x1, y1), (x2, y2), (0, 255, 0), 4)
                #
                # elif slope > 0.1: # slope -> _\
                #     right_lane.append((slope, intercept))
                #     cv2.line(warp, (x1, y1), (x2, y2), (0, 255, 0), 4)

    cv2.imshow('warp', warp)
    cv2.waitKey(0)

    return image, left_lane, right_lane

image, left_lane, right_lane = prepare(image)

if left_lane:
    left_mean = np.mean(left_lane, axis=0)
    left_slope = left_mean[0]
    left_intercept = left_mean[1]

if right_lane:
    right_mean = np.mean(right_lane, axis=0)
    right_slope = right_mean[0]
    right_intercept = right_mean[1]

# def fit_poly(shape, leftx, lefty, rightx, righty):
#     left_a, left_b, left_c = np.polyfit(lefty, leftx, 2)
#     right_a,  right_b, right_c = np.polyfit(righty, rightx, 2)
#     y = np.linspace(shape[0], shape[1]-1, shape[2])
#     left_x = left_a * y**2 + left_b * y + left_c
#     right_x = right_a * y**2 + right_b * y + right_c
#     return left_x, right_x, y


def find_lanes(image, drop_out=True):
    lane_lists = []
    if left_lane or right_lane:
        if drop_out:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < minpix:
                    cv2.drawContours(image=image, contours=[contour], contourIdx=-1, color=(0), thickness=-1)
        if left_lane:
            lane_lists.append(left_mean)

        if right_lane:
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
            # y = slope * x + intercept
            # x = (y - intercept) / slope
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

    for i in range(number):
        low = image.shape[0] - win_height*(i+1)
        high = image.shape[0] - win_height*i
        left_left = left_current - margin
        left_right = left_current + margin
        right_left = right_current - margin
        right_right = right_current + margin

        cv2.rectangle(out_img, (left_left, low), (left_right, high), (0, 255, 0), 4)
        cv2.rectangle(out_img, (right_left, low), (right_right, high), (0, 255, 0), 4)

        # left_side = left_left
        # if left_left < 0:
        #     left_side = 0
        #
        # right_side = right_right
        # if right_right < 0:
        #     right_side = 0
        #
        # left_rectangle = image[low:high, left_side:left_right]
        # right_rectangle = image[low:high, right_left:right_side]
        #
        # left_contours = find_contours(left_rectangle, False)
        # right_contours = find_contours(right_rectangle, False)
        #
        # side = 'left'
        # for list in left_contours, right_contours:
        #     big_contours = []
        #     for contour in list:
        #         area = cv2.contourArea(contour)
        #         if area > minpix:
        #             big_contours.append(contour)
        #
        #     if len(big_contours) > 1:
        #         if side == 'left':
        #             noise = big_contours[0]
        #             addition = np.zeros_like(noise)
        #             addition[:, :, 1:] = low
        #             addition[:, :, :1] = left_left
        #             fill = np.add(noise, addition)
        #             cv2.fillPoly(copy, [fill], (120, 0, 0))
        #
        #         elif side == 'right':
        #             noise = big_contours[-1]
        #             addition = np.zeros_like(noise)
        #             addition[:, :, 1:] = low
        #             addition[:, :, :1] = right_left
        #             fill = np.add(noise, addition)
        #             cv2.fillPoly(copy, [fill], (120, 0, 0))
        #
        #         small_nonzero = np.nonzero(copy)
        #         smallnonzeroy = np.array(small_nonzero[0])
        #         smallnonzerox = np.array(small_nonzero[1])
        #
        #     side = 'right'

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
                # left_current = 0
                pass
        if len(right_nonzero) > minpix:
            right_current = int(np.mean(nonzerox[right_nonzero]))
        else:
            try:
                right_current = int((low - right_intercept) / right_slope)
                print('follow right line')
            except NameError:
                # right_current = width
                pass
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
        # leftx0 = -rightx0 + width
        # try:
        #     if left_slope > 0:
        #         leftx0 = rightx0 - width // 2
        # except NameError:
        #     pass


    if len(rightx0) == 0:
        rightx0 = leftx0 + width // 2
        righty0 = lefty0
        # rightx0 = -leftx0 + width
        # try:
        #     if right_slope < 0:
        #         rightx0 = leftx0 + width // 2
        # except NameError:
        #     pass

    # if len(leftx0) == 0:
    #     try:
    #         if left_slope < 0:
    #             leftx0 = -rightx0 + width
    #         else:
    #             leftx0 = rightx0 - width // 2
    #     except NameError:
    #         leftx0 = rightx0 - width // 2
    #     lefty0 = righty0
    #
    # if len(rightx0) == 0:
    #     try:
    #         if right_slope > 0:
    #             rightx0 = -leftx0 + width
    #         else:
    #             rightx0 = leftx0 + width // 2
    #     except NameError:
    #         rightx0 = leftx0 + width // 2
    #     righty0 = lefty0

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

copy = np.copy(image)
leftx1, lefty1, rightx1, righty1, out_img1 = find_lanes(copy, True)

# 70/145
# if len(leftx1)<400 or len(rightx1)<400:
#     leftx, lefty, rightx, righty, out_img = find_lanes(image, False)
# else:
leftx, lefty, rightx, righty, out_img = leftx1, lefty1, rightx1, righty1, out_img1


def find_lanes_perspective():
    global M_inv
    zeros = np.zeros_like(frame)
    zeros[lefty, leftx] = 255
    zeros[righty, rightx] = 255
    _, M_inv = warp_perspective(frame, from_=dst, to=src)
    t_out_img = cv2.warpPerspective(zeros, M_inv, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)

    t_leftx = t_out_img[:, :width//2].nonzero()[1]
    t_lefty = t_out_img[:, :width//2].nonzero()[0]
    t_rightx = t_out_img[:, width//2:].nonzero()[1]+width//2
    t_righty = t_out_img[:, width//2:].nonzero()[0]

    if len(t_leftx)==0:
        t_leftx = -t_rightx + width
        t_lefty = t_righty

    if len(t_rightx) == 0:
        t_rightx = -t_leftx + width
        t_righty = t_lefty

    return t_leftx, t_lefty, t_rightx, t_righty, t_out_img

t_leftx, t_lefty, t_rightx, t_righty, t_out_img = find_lanes_perspective()


def fit_poly(leftx, lefty, rightx, righty):
    left_curve = np.polyfit(lefty, leftx, 2)
    right_curve = np.polyfit(righty, rightx, 2)

    return left_curve, right_curve


def visualise(image, low, high, num, left_curve, right_curve, plot):
    y = np.linspace(low, high, num)
    left_x = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
    right_x = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]
    high = np.append(np.arange(number)*win_height, image.shape[0]-1).reshape((-1, 1))

    for arr in left_x, right_x:
        arr = arr.astype(int)
        x = arr[high]
        con = np.concatenate((x, high), axis=1)

        cv2.polylines(image, [con], isClosed=False, color=(0, 255, 255), thickness=4)

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(leftx, -lefty, c='g')
        ax.scatter(rightx, -righty, c='r')
        ax.plot(left_x, -y, c='b')
        ax.plot(right_x, -y, c='b')
        plt.show()

    return image, left_x, right_x, y

print(out_img.shape[0])
left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
out_img, left_x, right_x, y = visualise(out_img, 0, height-1, height, left_curve, right_curve, False)

t_left_curve, t_right_curve = fit_poly(t_lefty, t_leftx, t_righty, t_rightx)
# t_out_img, t_left_x, t_right_x, _ = visualise(t_out_img, t_left_curve, t_right_curve)


# left_curve, right_curve = fit_poly(lefty, leftx, righty, rightx)
# t_left_curve, t_right_curve = fit_poly(t_lefty, t_leftx, t_righty, t_rightx)
# left_curve = np.polyfit(lefty, leftx, 2)
# right_curve = np.polyfit(righty, rightx, 2)
# y = np.linspace(0, image.shape[0] - 1, image.shape[0])
#
# left_x = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
# right_x = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]
#
# high = np.linspace(0, height-1, height).astype(int).reshape((-1, 1))

# for arr in left_x, right_x:
#     arr = arr.astype(int)
#     x = arr[high]
#     con = np.concatenate((x, high), axis=1)
#
#     # # Polilinia
#     cv2.polylines(out_img, [con], isClosed=False, color=(0, 0, 255), thickness=4)

# y = np.linspace(0, image.shape[0] - 1, image.shape[0])
# left_x = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
# right_x = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]
#
# y_eval = np.max(y)
# high = (np.arange(number)*win_height).reshape((-1, 1))
# left_rad = abs(((1 + (2 * left_curve[0] * y_eval + left_curve[1]) ** 2) ** 1.5) / (2 * left_curve[0]))
# right_rad = abs(((1 + (2 * right_curve[0] * y_eval + right_curve[1]) ** 2) ** 1.5) / (2 * right_curve[0]))
#
# angle = np.array([])
# for arr in left_x, right_x:
#     arr = arr.astype(int)
#     x = arr[high]
#     con = np.concatenate((x, high), axis=1)
#
#     # # Polilinia
#     cv2.polylines(out_img, [con], isClosed=False, color=(0, 255, 255), thickness=4)
#     cv2.imwrite('Test_frames/polylines.jpg', out_img)
#
#     # Linia z punktów na każdym pikselu
#     # for i, a in enumerate(arr):
#     #     cv2.circle(out_img, (a, int(y[i])), radius=1, color=(0, 255, 255), thickness=-1)
#
#     # Punkty na granicach okien
#     arr = np.array([])
#     for i in range(number):
#         x_1, y_1 = con[i]
#         cv2.circle(out_img, (x_1, y_1), radius=7, color=(255, 255), thickness=-1)

#         # Dyskretyzacja
#         if i is not number:
#             x_delta = int(x[i+1] - x[i])
#             if x_delta == 0:
#                 alfa = win_height
#             else:
#                 alfa = math.atan(win_height/x_delta)
#             arr = np.append(arr, alfa)
#             cv2.line(out_img, (x_1, y_1), (x_1 + x_delta, y_1), color=(255, 0, 0), thickness=3)
#             cv2.line(out_img, (x_1 + x_delta, y_1), (x_1 + x_delta, y_1 + win_height), color=(255, 255, 0), thickness=3)
#             cv2.putText(out_img, f'{round(alfa, 2)}', (x_1 + x_delta + 50, y_1 + win_height//2),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#
#     angle = np.append(angle, np.mean(arr))
#
# angle = np.max(np.absolute(angle))
#
# # left turn
# if left_rad < right_rad:
#     angle = angle
#
# # right turn
# else:
#     angle = -angle
#
# rotation_angle = round((90-abs(angle*180/math.pi)), 2)

# labels = ['Wycięcie fragmentu zdjęcia', 'Zmiana perspektywy', 'Skala szarości', 'Progowanie']
# imgs = [box, warp, gray, image]
# _, axs = plt.subplots(2, 2, figsize=(12, 12))
# axs = axs.flatten()
# i = 0
# for img, ax in zip(imgs, axs):
#     if len(img.shape) == 3:
#         img = np.flip(img, axis=-1)
#         ax.imshow(img)
#     else:
#         ax.imshow(img, cmap='gray')
#     ax.set_xlabel(labels[i], fontsize=16)
#     ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
#     i += 1

# fig = plt.figure()
# ax = plt.subplot()
# ax.scatter(leftx, -lefty, c='g')
# ax.scatter(rightx, -righty, c='r')
# ax.plot(left_x, -y, c='b')
# ax.plot(right_x, -y, c='b')
# plt.show()

def visualise_perspective(frame):
    poly = np.dstack((image, image, image)) * 255
    left = np.array([np.transpose(np.vstack([left_x, y]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_x, y])))])
    points = np.hstack((left, right))

    poly = cv2.fillPoly(poly, np.int_(points), (0, 255, 0))
    poly = cv2.warpPerspective(poly, M_inv, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    frame = cv2.addWeighted(frame, 1, poly, 0.6, 0)

    return frame


frame = visualise_perspective(frame)
# h = 25
#
# xl = 3*width//8
# yl = 4*height//5
#
# xl_1 = xl + h/math.tan(angle)
# yl_1 = yl + h
# xl_2 = xl - h/math.tan(angle)
# yl_2 = yl - h
#
# xl_1, yl_1, xl_2, yl_2 = map(int, (xl_1, yl_1, xl_2, yl_2))
#
# xr = 5*width//8
# yr = yl
#
# xr_1 = xr + h/math.tan(angle)
# yr_1 = yr + h
# xr_2 = xr - h/math.tan(angle)
# yr_2 = yr - h
#
# xr_1, yr_1, xr_2, yr_2 = map(int, (xr_1, yr_1, xr_2, yr_2))
#
# text = f'angle = {rotation_angle} deg'
#
# cv2.line(frame, (xl, yl), (xr, yr), color=(0, 0, 0), thickness=3)
# cv2.line(frame, (xl_1, yl_1), (xl_2, yl_2), color=(0, 0, 255), thickness=3)
# cv2.line(frame, (xr_1, yr_1), (xr_2, yr_2), color=(0, 0, 255), thickness=3)
# textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]//2
# cv2.putText(frame, text, (frame.shape[1]//2 - textsize, 250), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2,
#                     cv2.LINE_AA)
cv2.imshow('t_out_img', t_out_img)
cv2.waitKey(0)
cv2.imshow('out_img', out_img)
cv2.waitKey(0)
cv2.imshow('frame', frame)
cv2.waitKey(0)