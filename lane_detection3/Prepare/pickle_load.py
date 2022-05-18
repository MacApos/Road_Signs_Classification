import os
import cv2
import pickle
from imutils import paths
from datetime import datetime

path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
data_path = os.path.join(path, 'data')
labels_path = r'F:\krzysztof\PycharmProjects\Road_Signs_Classification\lane_detection3\Pickles\labels.p'

data_list = list(paths.list_images(data_path))
labels_list = pickle.load(open(labels_path, 'rb' ))

# file0 = open(path, 'rb')
# lines_dict0 = pickle.load(file0)
# file0.close()

new_labels = labels_list[:2547]
print(new_labels[0])

