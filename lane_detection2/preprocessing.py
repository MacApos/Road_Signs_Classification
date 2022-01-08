import cv2
import numpy as np


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def warp(image, draw_lines=False):
    height = image.shape[0]
    width = image.shape[1]

    #             szerokość wysokość
    src = np.float32([(550, 460),
                      (150, 720),
                      (1200, 720),
                      (770, 460)])

    dst = np.float32([(100, 0),
                      (100, 720),
                      (1100, 720),
                      (1100, 0)])

    if draw_lines:
        copy = np.copy(image)
        for i in range(src.shape[0]):
            x = src[i][0]
            y = src[i][1]
            if not i == src.shape[0]-1:
                x_1 = src[i+1][0]
                y_1 = src[i+1][1]
            else:
                x_1 = src[0][0]
                y_1 = src[0][1]
            copy = cv2.circle(image, (x, y), radius=0, color=(0, 0, 255), thickness=10)
            line = cv2.line(copy, (x, y), (x_1, y_1), color=(0, 255, 0), thickness=3)
        cv2.imshow('outout', line)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    output_top = cv2.warpPerspective(gray, M, (width, height), flags=cv2.INTER_LINEAR)
    output_front = cv2.warpPerspective(gray, M_inv, (width, height), flags=cv2.INTER_LINEAR)

    return output_top, output_front


def threshold(image):
    ret, image = cv2.threshold(image, 220, 225, cv2.THRESH_BINARY)
    if not ret:
        print('Error in threshold value')
    else:
        return image


image = cv2.imread('straight_lines2.jpg')
output_top, output_front = warp(image)
threshold = threshold(output_top)
# display = np.hstack((output_top, output_front))
cv2.imwrite('threshold.png', threshold)
cv2.imshow('output', threshold)
with open('threshold.npy', 'wb') as file:
    np.save(file, threshold)
cv2.waitKey(0)