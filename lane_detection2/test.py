import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

image = cv2.imread('test/test3.jpg')


def warp(image, inv=False):
    height = image.shape[0]
    width = image.shape[1]

    src = np.float32([(550, 460),
                      (150, 720),
                      (1200, 720),
                      (770, 460)])

    dst = np.float32([(100, 0),
                      (100, 720),
                      (1100, 720),
                      (1100, 0)])
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)

    M_inv = cv2.getPerspectiveTransform(dst, src)
    warp_inv = cv2.warpPerspective(image, M_inv, (width, height), flags=cv2.INTER_LINEAR)

    if inv:
        return warp_inv

    return warp, warp_inv


def threshold(image):
    ret, image = cv2.threshold(image, 150, 225, cv2.THRESH_BINARY)
    if not ret:
        "Invalid threshold value."
    else:
        return image


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lines(arr):
    copy = np.copy(image)
    for i in range(arr.shape[0]):
        x = int(arr[i][0])
        y = int(arr[i][1])
        x_1 = int(arr[i - 1][0])
        y_1 = int(arr[i - 1][1])
        circle = cv2.circle(copy, (x, y), radius=1, color=(0, 0, 255), thickness=10)
        line = cv2.line(circle, (x, y), (x_1, y_1), color=(0, 255, 0), thickness=3)
        text = cv2.putText(line, f'{x}, {y}', (x - 100, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2,
                           cv2.LINE_AA)
    return text


warp, warp_inv = warp(image)
gray = gray(warp)
image = threshold(gray)


def to_csv(arr, name):
    df = pd.DataFrame(arr)
    path = os.path.join('Arrays', name)
    df.to_csv(path, sep='\t', index=False, header=False)


# h = int(image.shape[0] * 0.1)
# w = int(image.shape[1] * 0.1)
# img = cv2.resize(image, (w, h))
# to_csv(img, 'img')


def find_lanes(image):
    histogram = np.sum(image[image.shape[0]//2:, :], axis=0)
    out_img = np.dstack((image, image, image))*255


    midpoint = int(histogram.shape[0]//2)
    left = np.argmax(histogram[:midpoint])
    right = midpoint + np.argmax(histogram[midpoint:])

    left_current = left
    right_current = right

    left_idx = []
    right_idx = []

    # Windows
    number = 9
    minpix = 50
    margin = 75
    height = int(image.shape[0]//number)

    nonzero = np.nonzero(image)
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    to_csv(nonzerox, 'nonzerox')
    to_csv(nonzeroy, 'nonzeroy')

    for i in range(number):
        low = image.shape[0] - height*(i+1)
        high = image.shape[0] - height*i
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
        if len(right_nonzero) > minpix:
            right_current = int(np.mean(nonzerox[right_nonzero]))

        to_csv(left_nonzero, 'left_nonzero')
        to_csv(right_nonzero, 'right_nonzero')
        to_csv(left_idx, 'left_idx')
        to_csv(right_idx, 'right_idx')

    try:
        left_idx = np.concatenate(left_idx)
        right_idx = np.concatenate(right_idx)
        print(right_idx.shape)
    except AttributeError:
        pass

    to_csv(left_idx, 'left_idx2')
    to_csv(right_idx, 'right_idx2')

    leftx = nonzerox[left_idx]
    lefty = nonzeroy[left_idx]
    rightx = nonzerox[right_idx]
    righty = nonzeroy[right_idx]

    return leftx, lefty, rightx, righty, out_img


# leftx, lefty, rightx, righty, out_img = find_lanes(image)


def fit_poly(shape, leftx, lefty, rightx, righty):
    left_a, left_b, left_c = np.polyfit(lefty, leftx, 2)
    right_a,  right_b, right_c = np.polyfit(righty, rightx, 2)
    y = np.linspace(0, shape[0]-1, shape[0])
    left_x = left_a * y**2 + left_b * y + left_c
    right_x = right_a * y**2 + right_b * y + right_c
    return left_x, right_x, y


margin = 100
nonzero = image.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx, lefty, rightx, righty, out_img = find_lanes(image)

# if (len(leftx)==0 or len(rightx)==0) or (len(rightx)==0 or len(righty==0)):
#     left_rad = 0
#     right_rad = 0
#     print('0')
#
# else:
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

left_idx = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[0] - margin)) &
            (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[0] + margin)))
print(nonzerox[left_idx])
right_idx = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[0] - margin)) &
             (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[0] + margin)))

# leftx = nonzerox[left_idx]
# lefty = nonzeroy[left_idx]
# rightx = nonzerox[right_idx]
# righty = nonzeroy[right_idx]

left_x, right_x, y = fit_poly(image.shape, leftx, lefty, rightx, righty)
print(nonzeroy)
to_csv(nonzeroy, 'nonzeroy')
left_left_poly = left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[0]
right_left_poly = left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[0]

left_right_poly = right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[0]
right_right_poly = right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[0]

fig = plt.figure()
ax = plt.subplot()
ax.scatter(leftx, -lefty, c='g')
ax.scatter(rightx, -righty, c='r')
ax.plot(left_x, -y, c='y')
ax.plot(right_x, -y, c='y')
ax.plot(left_left_poly, -nonzeroy, c='b')
ax.plot(right_left_poly, -nonzeroy, c='b')
ax.plot(left_right_poly, -nonzeroy, c='m')
ax.plot(right_right_poly, -nonzeroy, c='m')
plt.show()

cv2.imshow('image', out_img)
cv2.imwrite('test/threshold.png', out_img)
cv2.waitKey(0)
