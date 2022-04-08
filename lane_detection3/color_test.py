import cv2

frame5 = cv2.imread(r'F:\Nowy folder\10\Praca\Datasets\Video5\batch0\0.jpg')
frame6 = cv2.imread(r'F:\Nowy folder\10\Praca\Datasets\Video6\batch0\0.jpg')

cv2.imshow('frame5', frame5)
cv2.imshow('frame6', frame6)
cv2.waitKey(0)

name = 'frame5'
for frame in frame5, frame6:
    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    print(name, hls[1])

    hlv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    print(name, hlv.shape)

    name = 'frame6'