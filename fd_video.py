from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

# load Model Optimizer model
print("Loading Model...")
net = cv2.dnn.readNet('face-detection-adas-0001.xml',
                      'face-detection-adas-0001.bin')

# Specify target device.
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# initialize the video stream
print("Opening Video Stream from RPI Camera Module")
piCam = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream and send them to the
# Neural Network
while True:
    # get a frame from video stream
    frame = piCam.read()
    
    # resize it, because the Raspberry Pi is weak
    frame = imutils.resize(frame, width=400)
 
    # grab the frame dimensions and convert it to a blob
    # for the Neural Network to analyze
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
 
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence, default = 0.5
        if confidence < 0.5:
            continue
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
 
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        # show the output frame
        cv2.imshow("Frame", frame)

