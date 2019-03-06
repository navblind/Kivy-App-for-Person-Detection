from segment.sidewalkSeg import Segment
from direction.sidewalkDirection import procImage
from detection.objectDetection import Detect
from flask import Flask, request, send_file
import cv2
import base64

segObj = Segment("PSPNet", "512", "512", "model.ctpk")
dirObj = procImage()
detObj = Detect()

app = Flask(__name__)


@app.route("/")
def segment():
    img = request.args.get('img')
    img = base64.decodebytes(img)
    segImg = segObj.seg(img)
    direction = dirObj.proc(segImg)
    objects, newIm = detObj.run(img)
    hasPerson = True if len([False for i in objects if
                             i.split(":")[0] == 'person' and float(i.split(":")[1].strip()) > 0.9]) != 0 else False
    return direction + "\n" + hasPerson
