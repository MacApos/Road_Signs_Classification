import os
import cv2
import glob
import numpy as np

path = r'C:\Nowy folder\10\Praca\Datasets\camera_calibration\*.jpg'
images = glob.glob(path)
print(images)

# Kryteria algorytmu – wymagana zmiana parametrów między iteracjami, maks. liczba iteracji
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Przygotowanie współrzędnych 3D, z=0 – (0,0,0), (2,0,0), (2,0,0), ... (6,5,0)
# Wymiary szachownicy – 10x7
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Punkty na obiekcie w 3D
objpoints = []

# Punkty na zdjęciu w 2D
imgpoints = []

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    retval, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if retval:
        objpoints.append(objp)
        imgpoints.append(corners)

        corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, (9, 6), corners2, retval)
        # cv2.imshow('img', img)
        # cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread(images[3])
h, w = img.shape[:2]
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
print(roi)

# 1.
dst = cv2.undistort(img, mtx, dist, None, newcameramatrix)
cv2.imshow('dst', dst)
x, y, w, h = roi
res = dst[y:y+h, x:x+w]
print(dst.shape)
cv2.imshow('res', res)
cv2.waitKey(0)
