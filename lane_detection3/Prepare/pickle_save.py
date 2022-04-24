import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

path = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\small_test\*.jpg'
images = glob.glob(path)

road_images = []

for image in images[:10]:
    img = mpimg.imread(image)
    road_images.append(img)

