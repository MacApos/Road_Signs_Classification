import os
import cv2
import glob
import shutil
import pickle
import matplotlib.image as mpimg

dst = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\small_test'
src = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\test_set\clips\0531'

imglist = []
for i in os.listdir(src):
    path = os.path.join(src, i)
    filepath = glob.glob(path + '\\*.jpg')
    for file in filepath:
        imglist.append(file)
    # for j in os.listdir(path):
    #     filepath = os.path.join(path, j)
    #     imglist.append(filepath)

idx = 0
road_images = []
for img in imglist:
    name = f'\\{idx}.jpg'
    newfile = ''.join((dst, name))
    shutil.copy(img, newfile)
    image = mpimg.imread(img)
    road_images.append(image)
    idx += 1
    if idx == 100:
        break

pickle.dump(road_images, open(r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles'
                              r'\road_images.p', "wb"))
