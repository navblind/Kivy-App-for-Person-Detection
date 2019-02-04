import cv2
from detection.objectDetection import Detect

cap = cv2.VideoCapture(0)

personDetect = Detect()

while True:
    _, img = cap.read()

    found = personDetect.run(img)
    print(found) 

    if cv2.waitKey(1) >= 0:  # Break with ESC
        break
