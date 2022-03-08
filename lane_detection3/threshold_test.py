import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST\552.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
max_val = np.mean(np.amax(gray, axis=1)).astype(int)

thresh = cv2.threshold(img, 50, 225, cv2.THRESH_BINARY)
# otsu = cv2.threshold(gray, 0, 225, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)

plt.imshow(thresh)
plt.show()
# cv2.imshow('normal', normal)
cv2.imshow('gray', thresh)
cv2.waitKey(0)
