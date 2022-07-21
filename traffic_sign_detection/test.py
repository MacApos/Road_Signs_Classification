import os
import cv2
import numpy as np
from imutils import paths


path = 'archive/Train'
train_list = os.listdir(path)
class_list = [os.path.join(path, folder) for folder in train_list]

dir_size = 0
i = 0
img_size = []
images_path = []
for sign in class_list:
    image_path = list(paths.list_images(sign))
    dir_size += len(image_path)

    for image in image_path:
        images_path.append(image)
        img = cv2.imread(image)
        img_size.append(img.shape[:2])

    print(i)
    i += 1

max_val = max(img_size)
max_idx = img_size.index(max_val)

min_val = min(img_size)
min_idx = img_size.index(min_val)

images_path.pop(max_idx)
img_size.pop(max_idx)

max_val = max(img_size)
max_idx = img_size.index(max_val)

print(max_val, min_val)

for idx in max_idx, min_idx:
    img = cv2.imread(images_path[idx])
    print(images_path[idx])
    cv2.imshow('img', img)
    cv2.waitKey(0)