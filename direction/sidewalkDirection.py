import cv2
import numpy as np
from scipy.optimize import curve_fit


class procImage:
    vert = np.array([[0, 512], [128, 300], [384, 300], [512, 512]], np.int32)

    def proc(self, img, wantIm):
        cv2.imshow('original', img)
        cv2.waitKey()

        blur = self.erosion(img)
        blur = self.filterImg(blur)
        blur = self.median(blur)
        blur = self.closeDots(blur)
        blur = self.dilation(blur)

        edges = self.edgeDetect(blur)
        edges = self.roi(edges, [self.vert])

        lines = self.houghLines(edges)

        res = self.slopeAnalysis(lines, blur)
        return res

    def filterImg(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_pur = np.array([140, 0, 0])
        upper_pur = np.array([170, 255, 190])

        mask = cv2.inRange(hsv, lower_pur, upper_pur)
        res = cv2.bitwise_and(img, img, mask=mask)
        return res

    def roi(self, img, vert):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vert, 255)
        return cv2.bitwise_and(img, mask)

    def edgeDetect(self, img):
        edges = cv2.Canny(img, 250, 300)
        return cv2.GaussianBlur(edges, (3, 3), 0)

    def houghLines(self, edges):
        line = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, 7, 20)
        return line

    # Bluring options
    def bilateralFilter(self, img):
        return cv2.bilateralFilter(img, 9, 75, 75)

    def conv2D(self, img):
        kernel = np.ones((5, 5), np.float32) / 25
        return cv2.filter2D(img, -1, kernel)

    def blur(self, img):
        return cv2.blur(img, (5, 5))

    def gaussianBlur(self, img):
        return cv2.GaussianBlur(img, (5, 5), 0)

    def median(self, img):
        return cv2.medianBlur(img, 5)

    # Morphological transforations

    def erosion(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    def dilation(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)

    def removeDots(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def closeDots(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def removeSmallSpecs(self, img):
        pass

    # Thresholds
    def binaryInverse(self, img):
        pass

    def sobelX(self, img):
        pass

    def sobelY(self, img):
        pass

    # Slope related methods
    # TODO: Finish all direction analysis

    def slopeAnalysis(self, lines, blur):
        # Getting clusters
        l1x, l1y, l2x, l2y = self.cluster(lines, 512 / 2)

        # Checking number of clusters
        if (len(l1x) < 3 and len(l2x) < 3):
            return "unknown"
        elif (len(l1x) < 3):
            m, b = self.linearReg(l2x, l2y, blur)
            direction = self.numberOfLines(m, b)
            return direction
        elif (len(l2x) < 3):
            m, b = self.linearReg(l2x, l2y, blur)
            direction = self.numberOfLines(m, b)
            return direction
        else:

            m1, b1 = self.linearReg(l1x, l1y, blur)
            m2, b2 = self.linearReg(l2x, l2y, blur)

            # Check intersection
            if m1 < -2 or m1 > 0 or m2 > 2 or m2 < 0:
                intersect = self.linesIntersect(m1, b1, m2, b2)
                # Get direction if intersection is True
                if (intersect == True):
                    # TODO: DO polynomial reg and eval the curvature
                    equ1, polyx1, polyy1 = self.polyReg(l1x, l1y)
                    equ2, polyx2, polyy2 = self.polyReg(l2x, l2y)

                    direction = self.evalCurvature(polyx1, polyy1, polyx2, polyy2)
            else:
                intersect = False

            # Return logic
            if (intersect == False):
                return "straight"
            else:
                return direction

    def linearReg(self, X, Y, blur):
        mean_x = np.mean(X)
        mean_y = np.mean(Y)
        m = len(X)
        numer = 0
        denom = 0
        for i in range(m):
            numer += (X[i] - mean_x) * (Y[i] - mean_y)
            denom += (X[i] - mean_x) ** 2
        b1 = numer / denom
        b0 = mean_y - (b1 * mean_x)

        max_x = np.max(X) + 5
        min_x = np.min(X) - 5
        x = np.linspace(min_x, max_x, 1000)
        y = b0 + b1 * x

        l1inx1 = int((512 - b0) / b1)
        l1inx2 = int((200 - b0) / b1)
        cv2.line(blur, (int(l1inx1), 512), (int(l1inx2), 200), (255, 0, 0), 10)

        return b1, b0

    def func(self, x, a, b, c):
        return (a * (x ** 2)) + (b * x) + c

    def polyReg(self, xcors, ycors):
        time = np.array(xcors)
        avg = np.array(ycors)
        initialGuess = [5, 5, -.01]
        guessedFactors = [self.func(x, *initialGuess) for x in time]
        popt, pcov = curve_fit(self.func, time, avg, initialGuess)
        cont = np.linspace(min(time), max(time), 50)
        fittedData = [self.func(x, *popt) for x in cont]
        xcors = []
        ycors = []
        for count, i in enumerate(cont):
            xcors.append(i)
            ycors.append(fittedData[count])
        return popt, xcors, ycors

    def findCurvature(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        # return np.abs(d2y)/(1+ dy**2) **1.5
        return max(dx)

    def evalCurvature(self, x1, y1, x2, y2):
        # TODO: Eval the curvature of 2 quadratic lines for the direction of the sidewalk
        curve1 = self.findCurvature(x1, y1)
        curve2 = self.findCurvature(x2, y2)

        curve = curve1 - curve2

        if (curve > 0):
            return "right"
        else:
            return "left"

    def cluster(self, lines, xcord):
        l1x = []
        l1y = []
        l2x = []
        l2y = []
        if lines is not None:
            for i in lines:
                if (i[0][0] > xcord):
                    l2x.append(i[0][0])
                    l2y.append(i[0][1])
                    l2x.append(i[0][2])
                    l2y.append(i[0][3])
                else:
                    l1x.append(i[0][0])
                    l1y.append(i[0][1])
                    l1x.append(i[0][2])
                    l1y.append(i[0][3])

        return l1x, l1y, l2x, l2y

    def numberOfLines(self, m, b):
        if m > 0:
            return "Right"
        else:
            return "Left"

    def linesIntersect(self, m1, b1, m2, b2):
        x = (b1 - b2) / (m2 - m1)
        y = m1 * x + b1

        if (x < 512 or x > 0):
            if (y < 512 or y > 0):
                return True
        else:
            return False

    def findLineEq(self, point1, point2):
        m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b = point1[1] - m * point1[0]
        return m, b


if __name__ == "__main__":
    obj = procImage()
    # img = cv2.imread("test19_pred.png")
    img = cv2.imread("park2_pred.png")
    res = obj.proc(img, False)
    print(res)
