import pickle
import cv2
from datetime import datetime


lines0 = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\lane_labels_2.p'
lines1 = r'C:\Users\macie\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\train_line.p'

labels = pickle.load(open(lines0, "rb" ))

pickle.dump(labels, open('Pickles\lane_labels_2.p', "wb"))

file0 = open(lines0, 'rb')
lines_dict0 = pickle.load(file0)
file0.close()

file1 = open(lines1, 'rb')
lines_dict1 = pickle.load(file1)
file1.close()

print(lines_dict0[0].shape)
print(lines_dict1[0].shape)
