from keras.models import model_from_json
from imutils import paths
import numpy as np
import pickle

import cv2
import os

def find_file(path, ext):
    for file in os.listdir(path):
        if file.endswith(ext):
            if not file.startswith('warp'):
                normal = os.path.join(path, file)
            else:
                warp = os.path.join(path, file)

    return normal, warp

# path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
path = r'F:\krzysztof\Maciej_Apostol\StopienII\Video_data'
root_path = os.path.dirname(__file__)

dir_path = os.path.join(path, 'output')
validation_path = [os.path.join(dir_path, folder) for folder in os.listdir(dir_path)][-1]
test_path = os.path.join(path, 'test')
test_list = list(paths.list_images(test_path))

json_path, warp_json_path = find_file(validation_path, 'json')
weights_path, warp_weights_path = find_file(validation_path, 'h5')

json = open(json_path, 'r')
model = json.read()
json.close()
model = model_from_json(model)
model.load_weights(weights_path)

