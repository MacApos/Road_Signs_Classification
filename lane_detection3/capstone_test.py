import pickle
import os
import numpy as np
import cv2
# from moviepy.editor import VideoFileClip
from imutils import paths
from keras.models import load_model


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    small_img = cv2.resize(image, (160, 80))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255
    prediction = np.array(prediction, dtype='uint8')

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, prediction, blanks))


    blur = cv2.blur(lane_drawn, (5, 5))

    # Re-size to match the original image
    width = image.shape[1]
    height = image.shape[0]
    lane_image = cv2.resize(blur, (width, height))


    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


if __name__ == '__main__':
    # Load Keras model
    model = load_model('Pickles/full_CNN_model2.h5')
    # Create lanes object
    lanes = Lanes()

    path = r'C:\Nowy folder\10\Praca\Datasets\Video_data'
    test_list = list(paths.list_images(os.path.join(path, 'test')))
    test = [cv2.imread(i) for i in test_list]
    dir_path = os.path.join(path, 'output')

    output = []
    idx = 0
    for i in test[:60]:
        img = road_lines(i)
        if idx < 3:
            cv2.imwrite(os.path.join(dir_path, f'original{idx}.jpg'), img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output.append(img)
        idx += 1

    save_path = os.path.join(dir_path, 'original.gif')
    print(save_path)

    import imageio

    with imageio.get_writer(save_path, mode='I', fps=3) as writer:
        for image in output:
            print('saving')
            writer.append_data(image)
