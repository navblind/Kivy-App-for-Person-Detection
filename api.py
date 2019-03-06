from segment.sidewalkSeg import Segment
from direction.sidewalkDirection import procImage
from detection.objectDetection import Detect
from flask import Flask, request

segObj = Segment("MobileUNet", "512", "512", "model.ctpk")
dirObj = procImage()
detObj = Detect()

app = Flask(__name__)


@app.route("/")
def segment():
    img = request.args.get('img')
    segImg = segObj.seg(img)
    direction = dirObj.proc(segImg)
    objects, newIm = detObj.run(img)
    hasPerson = True if objects['person'] > 0.9 else False
    return direction + "\n" + hasPerson


if __name__ == "__main__":
    app.run(port=5000)
