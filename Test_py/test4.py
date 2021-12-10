import numpy as np
import cv2
from PIL import ImageTk, Image

image = cv2.imread(r'C:\Users\Maciej\Desktop\1.png')
image2 = Image.open(r'C:\Users\Maciej\Desktop\1.png')
# image = cv2.imread(r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\archive\Test\.png')

print(image.shape)
print(type(image2))

cv2.imshow('image', image)
cv2.waitKey(0)