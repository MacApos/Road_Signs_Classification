import os
import cv2

image = cv2.imread('Pictures/test3.jpg')
cv2.imshow('img', image)
cv2.waitKey(0)
name = 'test'
path = 'Pictures/'+name+'.jpg'

print(path)
cv2.imwrite(path, image)