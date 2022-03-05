import pickle
import cv2

filename = r'C:\Nowy folder\10\Praca\Datasets\full_CNN_train.p'
path = r'C:\Nowy folder\10\Praca\Datasets\TDS'

infile = open(filename, 'rb')
new_dict = pickle.load(infile)
infile.close()

for idx, val in enumerate(new_dict):
    dst = '\\'.join((path, f'{idx}.jpg'))
    print(dst)
    cv2.imwrite(dst, new_dict[idx])
