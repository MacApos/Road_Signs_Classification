import pickle

import numpy as np
import cv2
# from moviepy.editor import VideoFileClip
from IPython.display import HTML
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
    cv2.imshow('result', result)
    cv2.waitKey(0)

    # return result


if __name__ == '__main__':
    # Load Keras model
    model = load_model('full_CNN_model2.h5')
    # Create lanes object
    lanes = Lanes()

    test_list = list(paths.list_images(r'C:\Nowy folder\10\Praca\Datasets\Video_data\test'))
    test = [cv2.imread(i) for i in test_list]

    for i in test:
        res = road_lines(i)


    #
    # # Where to save the output video
    # vid_output = 'proj_reg_vid.mp4'
    # # Location of the input video
    # clip1 = VideoFileClip("project_video.mp4")
    # # Create the clip
    # vid_clip = clip1.fl_image(road_lines)
    # vid_clip.write_videofile(vid_output, audio=False)