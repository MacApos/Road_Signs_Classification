import os
import cv2
import glob
import random
import numpy as np

# nx = 7
# ny = 7
#
# original = cv2.imread("Chess.png")
# img = np.copy(original)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # ret to flaga, corners to znalezione rogi
# ret, corners = cv2.findChessboardCorners(gray, patternSize=(nx, ny), corners=None)
#
# # return flag
# if ret:
#     cv2.drawChessboardCorners(img, patternSize=(nx, ny), corners=corners, patternWasFound=ret)

fname = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection2\camera_cal\*.jpg'

# def pointExtractor(fname):
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# lokalizacja narożników irl
objpoints = []
# właściwa lokalizacja naróżników po odkształceniu
imgpoints = []

images = glob.glob(fname)

for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, (9, 6), corners, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

    # return objpoints, imgpoints


# def camerCalibration(objpoints, imgpoints, image):
image = cv2.imread('camera_cal/calibration11.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
shape = (image.shape[0], image.shape[1])

ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
undistorted = cv2.undistort(image, mtx, dist, None, mtx)
# return undistorted

cv2.imshow('image', undistorted)
cv2.waitKey(0)


# images = [os.path.join('camera_cal', img) for img in os.listdir('camera_cal')]
# print(images)
# fname = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection2\camera_cal\*.jpg'
# objpoints, imgpoints = pointExtractor(fname)
# image = cv2.imread('camera_cal/calibration11.jpg')
# output = camerCalibration(objpoints, imgpoints, image)
# cv2.imshow('image', undistorted)
# cv2.imshow('output', output)
# cv2.waitKey(0)
