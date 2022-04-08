import pickle
import cv2

filename = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\src.p'
path = r'C:\Nowy folder\10\Praca\Datasets\TDS'

infile = open(filename, 'rb')
new_dict = pickle.load(infile)
infile.close()
print(new_dict)

# for idx, val in enumerate(new_dict[2]):
#     dst = '\\'.join((path, f'{idx}.jpg'))
#     cv2.imwrite(dst, new_dict[idx])
