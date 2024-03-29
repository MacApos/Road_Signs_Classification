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
random = random.randint(0, len(list_dir)-1)
print(random)
image = cv2.imread(os.path.join(path, f'{3400}.jpg'))
image = cv2.resize(image, (1280, 720))
image = cv2.flip(image, 1)
frame = image

height = image.shape[0]
width = image.shape[1]

# src = np.float32([(550, 460),
#                   (150, 720),
#                   (1200, 720),
#                   (770, 460)])
#
# dst = np.float32([(100, 0),
#                   (100, 720),
#                   (1100, 720),
#                   (1100, 0)])

src = np.float32([(0, 720),
                  (400, 300),
                  (850, 300),
                  (1280, 720)])

dst = np.float32([(src[0]),
                  (src[0][0], 0),
                  (src[-1][0], 0),
                  (src[-1])])


def warp(image, inv=False):
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)

    M_inv = cv2.getPerspectiveTransform(dst, src)
    warp_inv = cv2.warpPerspective(image, M_inv, (width, height), flags=cv2.INTER_LINEAR)

    if inv:
        return warp_inv

    return warp, M_inv


def threshold(image, T):
    _, image = cv2.threshold(image, T, 250, cv2.THRESH_BINARY)
    # (_, image) = cv2.threshold(image, 0, 225, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)
    return image


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lines(image, arr, point_color=(255, 0, 0), line_color=(0, 255, 0)):
    arr = arr.astype(int)
    copy = np.copy(image)
    for i in range(arr.shape[0]):
        x, y = arr[i][0], arr[i][1]
        x_0, y_0 = arr[i - 1][0], arr[i - 1][1]
        cv2.circle(copy, (x, y), radius=1, color=point_color, thickness=10)
        cv2.line(copy, (x, y), (x_0, y_0), color=line_color, thickness=3)
        cv2.putText(copy, f'{x}, {y}', (x - 100, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return copy


def find_contours(image, display=False):
    contours = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contours = list(imutils.grab_contours(contours))
    if display:
        zeros = np.zeros((image.shape[0], image.shape[1], 3))
        for contour in contours:
            cnts = cv2.drawContours(image=zeros, contours=[contour], contourIdx=-1, color=(255, 0, 0),
                                    thickness=2)
            cv2.imshow('contours', cnts)
        cv2.waitKey(0)
    return contours


def to_csv(arr, name):
    df = pd.DataFrame(arr)
    path = os.path.join('../../lane_detection2/Arrays', name)
    df.to_csv(path, sep='\t', index=False, header=False)


def to_jpg(img, name):
    path = os.path.join('../../lane_detection2/test', (name + '.jpg'))
    cv2.imwrite(path, img)


box = lines(image, src)
box = lines(box, dst, line_color=(0, 0, 255))
warp, M_inv = warp(image)
gray = gray(warp)
max_val = np.mean(np.amax(gray, axis=1)).astype(int)
if max_val > (255*0.7):
    max_val = int(max_val*0.7)

image = threshold(gray, max_val)
contours = find_contours(image, display=False)

blur = cv2.GaussianBlur(gray[int(height*0.8):], (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)
lines = cv2.HoughLinesP(canny, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

cv2.imshow('out_img', image)
cv2.waitKey(0)

left_lane = []
right_lane = []
zeros = np.zeros_like(frame)
for line in lines:
    if line is not None:
        x1, y1, x2, y2 = line.reshape(4)
        y1 = y1 + int(height * 0.8)
        y2 = y2 + int(height * 0.8)
        cv2.line(warp, (x1, y1), (x2, y2), (0, 255, 0), 4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0: # /_ <- slope
            left_lane.append((slope, intercept))

        else: # slope -> _\
            right_lane.append((slope, intercept))

left_mean = np.mean(left_lane, axis=0)
left_slope = left_mean[0]
left_intercept = left_mean[1]

right_mean = np.mean(right_lane, axis=0)
right_slope = right_mean[0]
right_intercept = right_mean[1]

cv2.imshow('warp', warp)
cv2.waitKey(0)

def fit_poly(shape, leftx, lefty, rightx, righty):
    left_a, left_b, left_c = np.polyfit(lefty, leftx, 2)
    right_a,  right_b, right_c = np.polyfit(righty, rightx, 2)
    y = np.linspace(0, shape[0]-1, shape[0])
    left_x = left_a * y**2 + left_b * y + left_c
    right_x = right_a * y**2 + right_b * y + right_c
    return left_x, right_x, y


number = 9
minpix = 50
margin = 100
win_height = int(image.shape[0]//number)


def find_lanes(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    out_img = np.dstack((image, image, image))*225

    midpoint = int(histogram.shape[0]//2)
    left = np.argmax(histogram[:midpoint])
    right = midpoint + np.argmax(histogram[midpoint:])

    left_current = left
    right_current = right

    left_idx = []
    right_idx = []

    nonzero = np.nonzero(image)
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    for i in range(number):
        low = image.shape[0] - win_height*(i+1)
        high = image.shape[0] - win_height*i
        left_left = left_current - margin
        left_right = left_current + margin
        right_left = right_current - margin
        right_right = right_current + margin

        left_side = left_left
        if left_left < 0:
            left_side = 0

        right_side = right_right
        if right_right < 0:
            right_side = 0

        left_rectangle = image[low:high, left_side:left_right]
        right_rectangle = image[low:high, right_left:right_side]

        left_contour = find_contours(left_rectangle, False)
        right_contour = find_contours(right_rectangle, False)



        # for list in left_contour, right_contour:
        for contour in left_contour:
            area = cv2.contourArea(contour)
            print(f'Original length: {len(left_contour)}')
            if area < minpix or area == 0:
                try:
                    left_contour.remove(contour)
                except DeprecationWarning:
                    pass



            print()

        # for contour in right_contour:
        #     area = cv2.contourArea(contour)
        #     print(area)
        #     if area < minpix:
        #         print(f'area = {area}')
        #         right_contour.remove(area)

        # print(type(right_contour))
        # print()


        cv2.rectangle(out_img, (left_left, low), (left_right, high), (0, 255, 0), 4)
        cv2.rectangle(out_img, (right_left, low), (right_right, high), (0, 255, 0), 4)

        for arr in left_mean, right_mean:
            # y = slope * x + intercept
            # x = (y - intercept) / slope
            slope = arr[0]
            intercept = arr[1]
            y1 = 0
            x1 = int((y1 - intercept) / slope)
            y2 = height
            x2 = int((y2 - intercept) / slope)
            cv2.line(out_img, (x1, y1), (x2, y2), (255, 0, 0), 4)

        left_nonzero = ((nonzeroy >= low) & (nonzeroy <= high) &
                        (nonzerox >= left_left) & (nonzerox <= left_right)).nonzero()[0]

        right_nonzero = ((nonzeroy >= low) & (nonzeroy <= high) &
                         (nonzerox >= right_left) & (nonzerox <= right_right)).nonzero()[0]
        to_csv(right_nonzero, f'right_nonzero_{i}')


        right_nonzero = ((nonzeroy >= low) & (nonzeroy <= high) &
                         (nonzerox >= right_left) & (nonzerox <= right_right)).nonzero()[0]

        to_csv(left_nonzero, f'left_nonzero_{i}')


        left_idx.append(left_nonzero)
        right_idx.append(right_nonzero)

        if len(left_nonzero) > minpix:
            left_current = int(np.mean(nonzerox[left_nonzero]))
        else:
            left_current = int((high - left_intercept) / left_slope)

        if len(right_nonzero) > minpix:
            right_current = int(np.mean(nonzerox[right_nonzero]))
        else:
            right_current = int((high - right_intercept) / right_slope)

    try:
        left_idx = np.concatenate(left_idx)
        right_idx = np.concatenate(right_idx)
    except AttributeError:
        pass

    leftx = nonzerox[left_idx]
    lefty = nonzeroy[left_idx]
    rightx = nonzerox[right_idx]
    righty = nonzeroy[right_idx]

    return leftx, lefty, rightx, righty, out_img


leftx, lefty, rightx, righty, out_img = find_lanes(image)

left_curve = np.polyfit(lefty, leftx, 2)
right_curve = np.polyfit(righty, rightx, 2)
y = np.linspace(0, image.shape[0] - 1, image.shape[0])
left_x = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
right_x = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]
# y_eval = np.max(y)
#
# high = np.append(np.arange(number)*win_height, height-1).reshape((-1, 1))
#
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
#     cv2.polylines(out_img, [con], isClosed=False, color=(0, 0, 255), thickness=4)
#
#     # Linia z punktów na każdym pikselu
#     # for i, a in enumerate(arr):
#     #     cv2.circle(out_img, (a, int(y[i])), radius=1, color=(0, 255, 255), thickness=-1)
#
#     # Punkty na granicach okien
#     arr = np.array([])
#     for i in range(number+1):
#         x_1, y_1 = con[i]
#         cv2.circle(out_img, (x_1, y_1), radius=7, color=(0, 255, 255), thickness=-1)
#
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

# imgs = [box, warp, gray, image]
# _, axs = plt.subplots(2, 2, figsize=(12, 12))
# axs = axs.flatten()
# for img, ax in zip(imgs, axs):
#     if len(img.shape) == 3:
#         img = np.flip(img, axis=-1)
#         ax.imshow(img)
#     else:
#         ax.imshow(img, cmap='gray')
#     ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
#
# fig = plt.figure()
# ax = plt.subplot()
# ax.scatter(leftx, -lefty, c='g')
# ax.scatter(rightx, -righty, c='r')
# ax.plot(left_x, -y, c='b')
# ax.plot(right_x, -y, c='b')
# plt.show()

poly = np.dstack((image, image, image)) * 255

left = np.array([np.transpose(np.vstack([left_x, y]))])
right = np.array([np.flipud(np.transpose(np.vstack([right_x, y])))])
points = np.hstack((left, right))
poly = cv2.fillPoly(poly, np.int_(points), (0, 255, 0))
print(left, right)
poly = cv2.warpPerspective(poly, M_inv, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
frame = cv2.addWeighted(frame, 1, poly, 0.6, 0)
#
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

cv2.imshow('frame', frame)
cv2.imshow('out_img', out_img)
cv2.waitKey(0)