import os
import shutil
import pickle

dst = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST'
src = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\test_set\clips\0531'

imglist = []
for i in os.listdir(src):
    path = os.path.join(src, i)
    for j in os.listdir(path):
        filepath = os.path.join(path, j)
        imglist.append(filepath)

for idx, val in enumerate(imglist):
    name = f'\\{idx}.jpg'
    newfile = ''.join((dst, name))
    shutil.copy(imglist[idx], newfile)

# pickle.dump()