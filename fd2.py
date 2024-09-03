import numpy as np
from imutils.video import VideoStream
import imutils
import time
import cv2 as cv

# load model
net = cv.dnn.readNet('face-detection-adas-0001.xml', 'face-detection-adas-0001.bin')

# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# initialize the video stream
print("Opening Video Stream from RPI Camera Module")
piCam = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream and send them to the
# Neural Network
while True:
    # get a frame from video stream
    frame = cv.imread('Melty.jpg')
 
    # convert it to a blob
    # for the Neural Network to analyze
    blob = cv.dnn.blobFromImage(frame, size = (672,384), ddepth=cv.CV_8U)
 
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    out = net.forward()
    
    # Draw detected faces on the frame.
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
                                
        if confidence > 0.5:
            cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
            
    # show the output frame
        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    
# do a bit of cleanup
cv.destroyAllWindows()
vs.stop()
