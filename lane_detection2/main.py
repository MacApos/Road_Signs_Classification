import cv2
import preprocessing
import camerCalibration
import laneDetection
from moviepy.editor import VideoFileClip


def calibrate():
    fname = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection2\camera_cal\*.jpg'
    objpoints, imgpoints = camerCalibration.poinEctractor(fname)
    return objpoints, imgpoints


def pipline(frame):
    image = frame

    objpoints, imgpoints = calibrate()
    frame = camerCalibration.camerCalibration(objpoints, imgpoints, frame)

    frame, invM = preprocessing.warp(frame)
    frame = preprocessing.gray(frame)
    frame = preprocessing.threshold(frame)

    frame, left_curverad, right_curverad = laneDetection.search_around_poly(frame)
    frame = cv2.warpPerspective(frame, invM, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    # nakładanie przetworzonego obrazu na orginalny
    frame = cv2.addWeighted(frame, 0.3, image, 0.7, 0)

    curvature = (left_curverad+right_curverad) / 2
    car_position = image.shape[1]

    center = (abs(car_position-curvature)*(3.7/650))/10
    curvature = 'Radius of curvature ' + str(round(curvature, 2))
    center = str(round(center, 3)) + 'm away from center'

    frame = cv2.putText(frame, curvature, (frame.shape[0]//2, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, center, (frame.shape[0]//2, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


frame = cv2.imread('test/test3.jpg')
image = pipline(frame)
cv2.imshow('image', image)
cv2.imwrite('test/Example.png', image)
cv2.waitKey(0)


# def debug_frame(file):
#     cap = cv2.VideoCapture(file)
#
#     if not cap.isOpened():
#         print('Error opening the file, check its format')
#
#     cap.set(1, 100)
#     res, frame = cap.read()
#     frame = pipline(frame)
#     cv2.imshow('frame', frame)
#     cv2.waitKey(10000)
#
#
# def processFrames(infile, outfile):
#     output = outfile
#     output = outfile
#     clip = VideoFileClip(infile)
#     processingClip = clip.fl_image(pipline)
#     processingClip.write_videofile(output, audio=False)
#
#
# def main(infile, outfile):
#     processFrames(infile, outfile)
#
#
# if __name__ == '__main__':
#     infile = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection2\dashcam_video_trim.mp4'
#     outfile = r'C:\Users\Maciej\PycharmProjects\Road_Signs_Classification\lane_detection2\dashcam_video_trim_output2.mp4'
#     main(infile, outfile)
