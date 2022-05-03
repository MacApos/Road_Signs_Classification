import cv2
import numpy as np

image = cv2.imread('../Test_frames/curvy_lines2.jpg')
copy = np.copy(image)

height = image.shape[0]
width = image.shape[1]

blur = cv2.GaussianBlur(image, (5, 5), 0)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

def sobel(image):
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_16S, dx=1, dy=0, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_16S, dx=0, dy=1, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


# grad = sobel(gray[400:650, :])
canny = cv2.Canny(gray[400:650, :], 50, 150)
lines = cv2.HoughLinesP(canny, 2, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=5)

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
                cv2.line(copy[400:650, :], (x1, y1), (x2, y2), (0, 255, 0), 4)
            else:
                right_lane.append((slope, intercept))
                cv2.line(copy[400:650, :], (x1, y1), (x2, y2), (0, 255, 0), 4)


cv2.imshow('copy', copy)
cv2.waitKey(0)
