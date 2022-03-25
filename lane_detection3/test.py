import cv2
import numpy as np
import imutils

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
# image = cv2.imread('Test_frames/test2.jpg')

src = np.float32([(0, 720),
                  (400, 300),
                  (850, 300),
                  (1280, 720)])

dst = np.float32([(src[0]),
                  (src[0][0], 0),
                  (src[-1][0], 0),
                  (src[-1])])

i = 3
image = cv2.imread(fr'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST\{i}.jpg')

height = image.shape[0]
width = image.shape[1]

warp, M_inv = warp(image)
gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
max_val = np.mean(np.amax(gray, axis=1)).astype(int)

if max_val > (255*0.75):
    max_val = int(max_val*0.75)

_, image = cv2.threshold(gray, max_val, 250, cv2.THRESH_BINARY)

contours = cv2.findContours(image=image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
contours = imutils.grab_contours(contours)
edges = np.zeros((image.shape[0], image.shape[1], 3))
for contour in contours:
    cv2.drawContours(image=edges, contours=[contour], contourIdx=-1, color=(255, 255, 255),
                     thickness=2)

cv2.imshow('contours', edges)
cv2.waitKey(0)

gray_rgb = np.dstack((image, image, image)) * 225
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 250:
        cv2.drawContours(image=gray_rgb, contours=[contour], contourIdx=-1, color=(0, 0, 0),
                                thickness=-1)

left_max = np.nonzero(gray_rgb[:, :width//2])[0].max()
right_max = np.nonzero(gray_rgb[:, width//2:])[0].max()
down = min(left_max, right_max)
print(down)

cv2.rectangle(gray_rgb, (-10, down-int(0.2*height)), (1290, down), (255, 0, 0), 4)

blur = cv2.GaussianBlur(image[down-int(0.2*height):down, :], (5, 5), 0)
canny = cv2.Canny(blur, 50, 150)
left_lines = cv2.HoughLinesP(canny[:, :width//2], 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=2)
right_lines = cv2.HoughLinesP(canny[:, width//2:], 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=2)

# cv2.imwrite(f'Test_frames/canny_{i}.jpg', canny)

left_lane = []
right_lane = []

if left_lines is not None and right_lines is not None:
    add = 0
    for side in left_lines, right_lines:
        for line in side:
            x1, y1, x2, y2 = line.reshape(4)
            y1 = y1+down-int(0.2*height)
            y2 = y2+down-int(0.2*height)
            x1 = x1 + add
            x2 = x2 + add
            cv2.line(gray_rgb, (x1, y1), (x2, y2), (255, 255, 0), 4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
        add = width//2

    #         if slope < 0: # /_ <- slope
    #             left_lane.append((slope, intercept))
    #
    #         else: # slope -> _\
    #             right_lane.append((slope, intercept))
    #
    # if left_lane:
    #     left_mean = np.mean(left_lane, axis=0)
    #     left_slope = left_mean[0]
    #     left_intercept = left_mean[1]
    #     if left_slope:
    #         y1 = 0
    #         x1 = int((y1 - left_intercept) / left_slope)
    #         y2 = height
    #         x2 = int((y2 - left_intercept) / left_slope)
    #         cv2.line(gray_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)
    #     else:
    #         pass
    #
    # if right_lane:
    #     right_mean = np.mean(right_lane, axis=0).astype(int)
    #     right_slope = right_mean[0]
    #     right_intercept = right_mean[1]
    #     if right_slope:
    #         y1 = 0
    #         x1 = int((y1 - right_intercept) / right_slope)
    #         y2 = height
    #         x2 = int((y2 - right_intercept) / right_slope)
    #         cv2.line(gray_rgb, (x1, y1), (x2, y2), (255, 0, 0), 4)
    #     else:
    #         pass
cv2.imshow('canny', canny)
cv2.imshow('image', gray_rgb)
cv2.waitKey(0)





