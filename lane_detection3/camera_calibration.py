import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

path = 'camera_cal\left*.jpg'
# path = 'camera_cal\*.jpg'
images = glob.glob(path)

# Kryteria algorytmu – wymagana zmiana parametrów między iteracjami, maks. liczba iteracji
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Wymiary szachownicy – 7x6
grid = (7, 6)
# grid = (9, 6)

# Przygotowanie współrzędnych 3D, z=0 – (0,0,0), (2,0,0), (2,0,0), ... (6,5,0)
objp = np.zeros((grid[1]*grid[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)

# Punkty na obiekcie w 3D
objpoints = []

# Punkty na zdjęciu w 2D
imgpoints = []

i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    retval, corners = cv2.findChessboardCorners(gray, grid, None)
    print(retval)

    if retval:
        objpoints.append(objp)
        imgpoints.append(corners)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, grid, corners2, retval)

        if i == 8:
            cv2.imwrite('Pictures/camer_points.jpg', img)
            print(fname, i)
            cv2.imshow('camera_points', img)
            cv2.waitKey(500)

        i += 1

img = cv2.imread('camera_cal/left12.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

h, w = img.shape[:2]
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# 1
dst = cv2.undistort(img, mtx, dist, None, mtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

cv2.imwrite('Pictures/no_distortion.jpg', img)
# cv2.imshow('img', img)
# cv2.waitKey(0)

cv2.imwrite('Pictures/distortion.jpg', dst)
# cv2.imshow('dst', dst)
# cv2.waitKey(0)

# 2
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramatrix, (w, h), 5)
# dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
# dst = dst[y:y+h, x:x+w]

plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(dst)
plt.show()

# pickle.dump(mtx, open(r'Pickles\mtx.p', 'wb'))
# pickle.dump(dist, open(r'Pickles\dist.p', 'wb'))