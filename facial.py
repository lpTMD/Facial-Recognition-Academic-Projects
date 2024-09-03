import cv2
from picamera import PiCamera

# Load the model.
net = cv2.dnn.readNet('face-detection-adas-0001.xml',
                     'face-detection-adas-0001.bin')
# Specify target device.
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# Read an image.
frame = cv2.imread('Melty.jpg')
if frame is None:
    raise Exception('Image not found!')

# Prepare input blob and perform an inference.
blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
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
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))
# Save the frame to an image file.
cv2.imwrite('out.png', frame)