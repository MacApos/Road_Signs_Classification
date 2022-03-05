import cv2
import numpy as np



def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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

    # src = np.float32([[200, 460], [1150, 460], [436, 220], [913, 220]])
    # dst = np.float32([[300, 720], [1000, 720], [400, 0], [1200, 0]])

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
            text = cv2.putText(copy, f'{x}, {y}', (int(x-100), int(y-15)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255),
                               2, cv2.LINE_AA)
            copy = cv2.circle(text, (x, y), radius=0, color=(0, 0, 255), thickness=10)
            line = cv2.line(copy, (x, y), (x_1, y_1), color=(0, 255, 0), thickness=3)
        cv2.imshow('outout', line)

    M = cv2.getPerspectiveTransform(src, dst)

    M_inv = cv2.getPerspectiveTransform(dst, src)

    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    # output_front = cv2.warpPerspective(image, M_inv, (width, height), flags=cv2.INTER_LINEAR)

    return warp, M_inv


def threshold(image):
    # ret, image = cv2.threshold(image, 170, 225, cv2.THRESH_BINARY)
    (ret, image) = cv2.threshold(image, 0, 225, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if not ret:
        print('Error in threshold value')
    else:
        return image


image = cv2.imread('test/test3.jpg')
image = cv2.flip(image, 1)
frame, inv_M = warp(image, True)
cv2.imshow('frame', frame)
gray = gray(frame)
threshold = threshold(gray)
cv2.imshow('threshold', threshold)
cv2.imwrite('test/threshold.png', threshold)
with open('test/threshold.npy', 'wb') as file:
    np.save(file, threshold)
cv2.waitKey(0)
