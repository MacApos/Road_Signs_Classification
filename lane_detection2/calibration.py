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

fname = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection\camera_cal\*.jpg'

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


def undistorted(image):
    shape = (image.shape[1], image.shape[0])
    ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    return undistorted


input = cv2.imread(r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection\camera_cal\calibration1.jpg')
output = undistorted(input)
display = np.hstack((input, output))
cv2.imshow('display', display)
cv2.waitKey(0)