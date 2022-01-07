import cv2
import numpy as np

src = np.float32([(550, 460),
                  (150, 720),
                  (1200, 720),
                  (770, 460)])

dst = np.float32([(100, 0),
                  (100, 720),
                  (1100, 720),
                  (1100, 0)])

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)


def front_to_top(img):
    size = (1200, 720)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)


def top_to_front(img):
    size = (1200, 720)
    return cv2.warpPerspective(img, M_inv, size, flags=cv2.INTER_LINEAR)


input = cv2.imread('straight_lines2.jpg')
output_top = front_to_top(input)
output_front = top_to_front(output_top)
# display = np.hstack((output_top, output_front))
cv2.imshow('outout', output_front)
cv2.waitKey(0)