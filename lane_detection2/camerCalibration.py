import cv2
import glob
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


def poinEctractor(fname):
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

    return objpoints, imgpoints


def camerCalibration(objpoints, imgpoints, image):
    shape = (image.shape[0], image.shape[1])
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    return undistorted


# fname = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection2\camera_cal\*.jpg'
# objpoints, imgpoints = poinEctractor(fname)
# image = cv2.imread(r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection2\camera_cal'
#                    r'\calibration1.jpg')
# output = camerCalibration(objpoints, imgpoints, image)
# display = np.hstack((image, output))
# cv2.imshow('display', display)
# cv2.waitKey(0)