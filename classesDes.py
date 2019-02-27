from segment.sidewalkSeg import Segment
from direction.sidewalkDirection import procImage
from detection.objectDetection import Detect

print("Module loading done")

#To make object for segmentation
#Parameters ->  modelName, width, height, checkpoint_path
#Note: All parameters must be strings

segObj = Segment("MobileUNet", "512", "512", "model.cptk")


#To make direction analaysis object

dirObj = procImage()

#To make object detection object

detObj = Detect()





## Calling methods

#Semantic Segmentation ->  object.seg(image)    | Parameter is an image  | Returns an image segmented image
#Direction analysis ->  object.proc(segmentatedImage)   | Paramater is an image that has undergone semantic segmentation | Returns a string for direction to move in
#Object detection ->  object.run(image)    | Parameter is an image | Returns all objects found with confidence factor. Also returns image with boxes drawn around objects


##Example 

_, img =cap.read()

segImg = segObj.seg(img)
direction = dirObj.proc(segImg)

objects, newIm = detObj.run(img)
