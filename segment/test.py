from sidewalkSeg import Segment
import cv2

obj = Segment("MobileUNet", "512", "512", "checkpoints/model.ckpt")

img = cv2.imread("test.jpg")

im = obj.seg(img)

cv2.imshow("im", im)
cv2.waitKey()
