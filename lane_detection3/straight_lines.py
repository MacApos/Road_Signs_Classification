import cv2
import numpy as np

image = cv2.imread(r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST\450.jpg')
height = image.shape[0]
width = image.shape[1]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)

src = np.array([(0, 720), (400, 300), (850, 300), (1280, 720)])
src = src.reshape((-1, src.shape[0], src.shape[1]))
mask = np.zeros_like(canny)
cv2.fillPoly(mask, src, 255)
masked_image = cv2.bitwise_and(canny, mask)

lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

cv2.imshow('res', lines)
cv2.waitKey(0)
#
# left_lane = []
# right_lane = []
# zeros = np.zeros_like(image)
# for line in lines:
#     if line is not None:
#         x1, y1, x2, y2 = line.reshape(4)
#         cv2.line(zeros, (x1, y1), (x2, y2), (0, 255, 0), 4)
#         print(x1, y1, x2, y2)
#         parameters = np.polyfit((x1, x2), (y1, y2), 1)
#         slope = parameters[0]
#         intercept = parameters[1]
#
#         if slope < 0: # /_ <- slope
#             left_lane.append((slope, intercept))
#
#         else: # slope -> _\
#             right_lane.append((slope, intercept))
#
# left_mean = np.mean(left_lane, axis=0)
# right_mean = np.mean(right_lane, axis=0)
#
# # visualization
# for arr in left_mean, right_mean:
#     # y = slope * x + intercept
#     # x = (y - intercept) / slope
#     slope = arr[0]
#     intercept = arr[1]
#     y1 = 0
#     x1 = int((y1 - intercept) / slope)
#     y2 = image.shape[0]
#     x2 = int((y2 - intercept) / slope)
#
#     cv2.line(zeros, (x1, y1), (x2, y2), (0, 255, 0), 4)
#
# mean = np.array([left_mean, right_mean])

# if mean is not None:
#     for x1, y1, x2, y2 in mean:
#         pass
        # print(x1, y1, x2, y2)


