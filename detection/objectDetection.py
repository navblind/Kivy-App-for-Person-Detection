# Import the neccesary libraries
import numpy as np
# import argparse
import cv2


class Detect:
    classNames = {0: 'background',
                  1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                  5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                  10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                  14: 'motorbike', 15: 'person', 16: 'pottedplant',
                  17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
    net = cv2.dnn.readNetFromCaffe("detection/MobileNetSSD_deploy.prototxt", "detection/MobileNetSSD_deploy.caffemodel")

    def run(self, frame):
        frame_resized = cv2.resize(frame, (300, 300))
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        self.net.setInput(blob)
        detections = self.net.forward()

        cols = frame_resized.shape[1]
        rows = frame_resized.shape[0]

        found = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                class_id = int(detections[0, 0, i, 1])

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                heightFactor = frame.shape[0] / 300.0
                widthFactor = frame.shape[1] / 300.0
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop = int(widthFactor * xRightTop)
                yRightTop = int(heightFactor * yRightTop)
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (255, 255, 0))

                if class_id in self.classNames:
                    label = self.classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                  (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                    found.append(label)
        return found, frame


if __name__ == "__main__":
    Detect().run()
