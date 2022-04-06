import os
import random
import cv2
import glob
import shutil
import pickle
import matplotlib.image as mpimg

src = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\train_set\clips'
dst = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
image = cv2.imread(r'C:\Nowy folder\10\Praca\Datasets\tu-simple\train_set\clips\0601\1495492800537479888\1.jpg')

# src = r'F:\Nowy folder\10\Praca\Datasets\tu-simple\train_set\clips'
# dst = r'F:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
# image = cv2.imread(r'F:\Nowy folder\10\Praca\Datasets\tu-simple\train_set\clips\0601\1495492800537479888\1.jpg')

def sort_files(path):
    path = sorted([int(i) for i in path])
    return [str(i) for i in path]

def randomize_file(path):
    return random.sample(path, len(path))


cv2.imshow('image', image)
cv2.waitKey(0)

imglist0 = [os.path.join(src, i) for i in os.listdir(src)]
print(imglist0)

imglist = []
for j in imglist0:
    j = os.listdir(j)
    print(j)
    # for k in os.listdir(j):
    #     l = os.path.join(j, k)
    #     print(l)
        # imglist2 = os.listdir(l)
        # random_img = random.randint(1, len(imglist2))
        # path = l + fr'\{random_img}.jpg'
        # imglist.append(path)

# idx = 0
# road_images = []
# for img in imglist:
#     name = f'\\{idx}.jpg'
#     newfile = ''.join((dst, name))
#     shutil.copy(img, newfile)
#     image = mpimg.imread(img)
#     road_images.append(image)
#     idx += 1
    # if idx == 1400:
    #     break
# #
# pickle.dump(road_images, open(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles'
#                               r'\road_images.p', "wb"))
