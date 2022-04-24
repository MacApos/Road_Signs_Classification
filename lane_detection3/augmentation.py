import os
import cv2
import pickle
import random
import numpy as np
from scipy import ndimage, misc
from skimage.transform import resize

def sort_path(path):
    sorted_path = []
    for file in os.listdir(path):
        number = int(''.join(n for n in file if n.isdigit()))
        sorted_path.append(number)

    sorted_path = sorted(sorted_path)
    return [path + fr'\{str(f)}.jpg' for f in sorted_path]


def rotate(path, i, angles):
    for image in path[:10]:
        angle = random.sample(angles, 1)[0]
        img = cv2.imread(image)

        rotated_img = ndimage.rotate(img, angle, reshape=False)
        cv2.imwrite(tmp_rotation + fr'\{i}.jpg', rotated_img)
        i += 1


def flip(path, i):
    for image in path[:10]:
        img = cv2.imread(image)

        flipped_img = np.fliplr(img)
        cv2.imwrite(tmp_flip + fr'\{i}.jpg', flipped_img)
        i += 1


road_images_path = r'F:\Nowy folder\10\Praca\Datasets\Video_data\frames'
labels_images_path = r'F:\Nowy folder\10\Praca\Datasets\Video_data\labels'

tmp_rotation = r'F:\Nowy folder\10\Praca\Datasets\Video_data\rotation'
tmp_flip = r'F:\Nowy folder\10\Praca\Datasets\Video_data\flip'

labels = pickle.load(open( "Pickles/lane_labels.p", "rb" ))
road_images = sort_path(road_images_path)
labels_images = sort_path(labels_images_path)

image = cv2.imread(road_images[0])
height = image.shape[0]
width = image.shape[1]

# coeff_list = []
#
# for label in labels:
#     coefficients = []
#     for coeff in label:
#         # print(coeff)
#         coefficients.append([coeff])
#     coeff_list.append(coefficients)

for val, idx in enumerate(road_images):
    image = cv2.imread(val)
    image = image[175:, :]
    scaled_image = resize(image, (height//8, width//8, 3))
    cv2.imwrite(tmp_flip + fr'\{idx}.jpg', scaled_image)

img_num = len(road_images)
angles_range = [i for i in range(-2, 3) if i != 0]

rotate_road_images = random.sample(road_images, int(img_num*0.3))
rotate_labels_images = random.sample(labels_images, int(img_num*0.3))

flip_road_images = random.sample(road_images, int(img_num*0.5))
flip_labels_images = random.sample(labels_images, int(img_num*0.5))



random = random.randint(0, len(road_images_path)-1)
print(random)

image = cv2.imread(os.path.join(road_images_path, f'{random}.jpg'))

cv2.imshow('image', image[175:, :])
cv2.waitKey(0)