import numpy as np
import cv2
import matplotlib.image as mpimg
import glob
import pickle
import re
from scipy.ndimage.interpolation import rotate


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def load_drawn_images():
    """Load re-drawn lane image locations"""
    drawn_image_locs = glob.glob('draw/*.jpg')
    sort_drawn_image_locs = sorted(drawn_image_locs, key=natural_key)
    return sort_drawn_image_locs


# def pipeline(img, R_thresh=(230, 255)):
#     """Threshold the re-drawn images for high red threshold"""
#     img = np.copy(img)
#     R = img[:, :, 0]
#
#     R_binary = np.zeros_like(R)
#     R_binary[(R >= R_thresh[0]) & (R <= R_thresh[1])] = 1
#
#     combined_binary = np.zeros_like(R_binary)
#     combined_binary[(R_binary == 1)] = 1
#     return combined_binary

src = np.float32([[290,650],
                  [570,525],
                  [710,525],
                  [990,650]])

dst = pts2 = np.float32([[0,720],
                         [0,0],
                         [1280,0],
                         [1280,720]])


def warp_perspective(image, from_, to):
    M = cv2.getPerspectiveTransform(from_, to)
    warp = cv2.warpPerspective(image, M, (1280, 720), flags=cv2.INTER_LINEAR)
    return warp, M


def gray_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def threshold(image, T):
    _, image = cv2.threshold(image, T, 250, cv2.THRESH_BINARY)
    return image


def pipeline(img, R_thresh=(230, 255)):
    warp, _ = warp_perspective(img, src, dst)
    gray = gray_img(warp)
    max_val = np.mean(np.amax(gray, axis=1)).astype(int)
    max_val = int(max_val*0.85)
    thresh = threshold(gray, max_val)
    return thresh


def left_line_detect(out_img, leftx_current, margin, minpix, nonzerox, nonzeroy, win_y_low, win_y_high, window_max,
                     counter):
    # Identify left window boundaries
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

    left_tracker = True
    if counter >= 5:
        if win_xleft_high > window_max:
            # or win_xleft_low < 0
            left_tracker = False

    return good_left_inds, leftx_current, left_tracker


def right_line_detect(out_img, rightx_current, margin, minpix, nonzerox, nonzeroy, win_y_low, win_y_high, window_max,
                      counter):
    # Identify right window boundaries
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin


    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)


    # Identify the nonzero pixels in x and y within the window
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    right_tracker = True
    if counter >= 5:
        if win_xright_high > window_max or win_xright_low < 0:
            right_tracker = False

    return good_right_inds, rightx_current, right_tracker


def lane_detection(image_list):
    """Iterates through each binary thresholded image. Uses sliding
    windows to detect lane points, and fits lines to these points.
    The polynomial coefficients of this line are then appended to
    the lane_labels list as those will be the training labels.
    The below code is a modified version of my computer vision-based
    Advanced Lane Lines project.
    """
    for fname in image_list:
        # Read image
        img = mpimg.imread(fname)
        # Binary threshold the image
        binary_warped = pipeline(img)

        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 225


        print(len(binary_warped.shape))
        cv2.imshow('out_img', out_img)
        cv2.waitKey(0)


        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 35
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        left_tracker = True
        right_tracker = True
        counter = 0

        # Step through the windows one by one
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            window_max = binary_warped.shape[1]



            if left_tracker == True and right_tracker == True:
                good_left_inds, leftx_current, left_tracker = left_line_detect(out_img, leftx_current, margin, minpix, nonzerox,
                                                                 nonzeroy, win_y_low, win_y_high, window_max, counter)
                good_right_inds, rightx_current, right_tracker = right_line_detect(out_img, rightx_current, margin, minpix, nonzerox,
                                                                    nonzeroy, win_y_low, win_y_high, window_max,
                                                                    counter)
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                counter += 1
            elif left_tracker == True:
                good_left_inds, leftx_current, left_tracker = left_line_detect(out_img, leftx_current, margin, minpix, nonzerox,
                                                                 nonzeroy, win_y_low, win_y_high, window_max, counter)
                # Append these indices to the list
                left_lane_inds.append(good_left_inds)
            elif right_tracker == True:
                good_right_inds, rightx_current, right_tracker = right_line_detect(out_img, rightx_current, margin, minpix, nonzerox,
                                                                    nonzeroy, win_y_low, win_y_high, window_max,
                                                                    counter)
                # Append these indices to the list
                right_lane_inds.append(good_right_inds)
            else:
                break

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            print(left_tracker, right_tracker)

        cv2.imshow('out_img', out_img)
        cv2.waitKey(0)

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Look around initial line to improve fit
        margin2 = 150
        left_lane_inds = (
                    (nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin2)) & (
                        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin2)))
        right_lane_inds = (
                    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin2)) & (
                        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin2)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Append to the labels list
        lane_labels.append(np.append(left_fit, right_fit))

        cv2.imshow('binary_warped', binary_warped)
        cv2.waitKey(0)


# Load in the re-drawn lane images
# images = load_drawn_images()

# Make a list to hold the 'labels' - six coefficients, two for each line
lane_labels = []

# Run through all the images
lane_detection([r'F:\Nowy folder\10\Praca\Datasets\Video_data\train_set\1300.jpg'])

# Save the final list to a pickle file for later
pickle.dump(lane_labels, open('lane_labels.p', "wb"))