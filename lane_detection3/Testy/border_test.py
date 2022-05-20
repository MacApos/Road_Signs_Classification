import cv2
import numpy as np

image = cv2.imread(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pictures\original.jpg')
zeros = np.zeros((260, image.shape[1], 3))

image = image[240:, :, :]
image = image[240:, :, :]

resized_img = cv2.copyMakeBorder(image, 20, 0, 0, 0, cv2.BORDER_REPLICATE)

print(resized_img.shape)

cv2.imshow('zeros', resized_img)
cv2.waitKey(0)