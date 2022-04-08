import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

data_path = r'F:\Nowy folder\10\Praca\Datasets\Video_data'
new_path = r'F:\Nowy folder\10\Praca\Datasets\Video_data\Video_frames'
name_dir = ['Video1', 'Video2', 'Video3', 'Video4',]

i = 0
for name in name_dir:
    path = os.path.join(data_path, name)

    list_dir = os.listdir(path)
    list_dir = [file.split('.jpg')[0] for file in list_dir]
    list_dir.sort(key = int)

    for file in list_dir:
        src = path + fr'\{file}.jpg'
        dst = new_path + fr'\{i}.jpg'
        os.rename(src, dst)
        i += 1