import os
import random
from imutils import paths

base = r'F:\Nowy folder\10\Praca\Datasets\Test\downloads'

dir_paths = [os.path.join(base, dir) for dir in os.listdir(base)]

for dir_path in dir_paths:
    i = 0
    print(dir_path)
    images = paths.list_images(dir_path)


    for img in images:
        fname = img.split('\\')
        print(fname)
        print(os.path.join(fname[1], fname[0], f'{random.randint(50, 500):04d}.jpg'))
        # os.rename(img, os.path.join(fname[]))
        i += 1