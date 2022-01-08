import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
cv2.namedWindow('trackbar_1',cv2.WINDOW_FREERATIO)
cv2.namedWindow('trackbar_2',cv2.WINDOW_FREERATIO)
cv2.namedWindow('trackbar_3',cv2.WINDOW_FREERATIO)
cv2.namedWindow('trackbar_4',cv2.WINDOW_FREERATIO)
class DetectShapes2(DetectShapes):
    def __init__(self, closed: bool = True, epsilon: float = 0.04, printResult: bool = False,WindName='trackbar',trackbar: bool = False) -> None:
        self.printResult = printResult
        self.closed = closed
        self.epsilon = epsilon
        self.WindName = WindName
        if trackbar:
            self.createTrackbar()

    def createTrackbar(self):
        cv2.createTrackbar("epsilon", self.WindName, int(self.epsilon*100), 1000, self.Epsilon)

    def Epsilon(self, value: int):
        if value < 0 or value > 1000:
            return
        self.epsilon = value/100

class Threshold2(Threshold):
    def __init__(self, lowerBound: int, upperBound: int, mode: int = cv2.THRESH_BINARY, showOutput: bool = False, outputWindowName: str = 'Threshold',trackbar: bool = False,namedWindow=True,WindName='trackbar') -> None:
        self.showOutput = showOutput
        self.outputWindowName = outputWindowName
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.mode = mode
        self.WindName = WindName
        if trackbar:
            self.createTrackbar()
        self.namedWindow = namedWindow

    def createTrackbar(self):
        #if self.namedWindow:
        cv2.namedWindow(self.WindName)
        cv2.createTrackbar("lower", self.WindName, self.lowerBound, 255, self.setLBH)
        cv2.createTrackbar("upper", self.WindName, self.upperBound, 255, self.setUBH)

    def setLBH(self, value: int):
        if value < 0 or value > 255:
            return
        self.lowerBound = value

    def setUBH(self, value: int):
        if value < 0 or value > 255:
            return
        self.upperBound = value

class ThresholdContours2(ThresholdContours):
    def __init__(self, minArea: float, maxArea: float, printDebug: bool = False,trackbar: bool = False,namedWindow=True,WindName='trackbar') -> None:
        self.minArea = minArea
        self.maxArea = maxArea
        self.printDebug = printDebug
        self.WindName = WindName
        if trackbar:
            self.createTrackbar()
        self.namedWindow = namedWindow

    def createTrackbar(self):
        #if self.namedWindow:
        cv2.namedWindow(self.WindName)
        cv2.createTrackbar("Conto_min", self.WindName, self.minArea, 100000, self.Conto_minR)
        cv2.createTrackbar("Conto_max", self.WindName, self.maxArea, 100000, self.Conto_maxR)

    def Conto_minR(self, value: int):
        if value < 0 or value > 100000:
            return
        self.minArea = value

    def Conto_maxR(self, value: int):
        if value < 0 or value > 100000:
            return
        self.maxArea = value

class HSVThreshold2(HSVThreshold):
    def __init__(self, lowerBound: np.array = np.array([0, 0, 0]), upperBound: np.array = np.array([255, 255, 255]), trackbar: bool = False, showOutput: bool = False, outputWindowName: str = 'HSVThreshold',WindName='trackbar') -> None:
        self.showOutput = showOutput
        self.outputWindowName = outputWindowName
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.WindName = WindName
        if trackbar:
            self.createTrackbar()

    def createTrackbar(self):
        cv2.namedWindow(self.WindName)
        cv2.createTrackbar("HSV_LBH", self.WindName, self.lowerBound[0], 255, self.setLBH)
        cv2.createTrackbar("HSV_UBH", self.WindName, self.upperBound[0], 255, self.setUBH)
        cv2.createTrackbar("HSV_LBS", self.WindName, self.lowerBound[1], 255, self.setLBS)
        cv2.createTrackbar("HSV_UBS", self.WindName, self.upperBound[1], 255, self.setUBS)
        cv2.createTrackbar("HSV_LBV", self.WindName, self.lowerBound[2], 255, self.setLBV)
        cv2.createTrackbar("HSV_UBV", self.WindName, self.upperBound[2], 255, self.setUBV)

class Erode(PostProcessingBlock):

    def __init__(self,kernel,iterations, showOutput = False, outputWindowName = 'Erosion') -> None:
        self.showOutput = showOutput
        self.outputWindowName = outputWindowName
        self.kernel = kernel
        self.iterations = iterations
        

    def run(self, input):
        output = input.copy()
        cv2.erode(output,kernel = self.kernel, iterations=self.iterations)
        if self.showOutput:
            cv2.imshow(self.outputWindowName, output)
        return output
class Dilate(PostProcessingBlock):
    def __init__(self,kernel,iterations, showOutput = False, outputWindowName = 'Erosion') -> None:
        self.showOutput = showOutput
        self.outputWindowName = outputWindowName
        self.kernel = kernel
        self.iterations = iterations
        

    def run(self, input):
        output = input.copy()
        cv2.dilate(output,kernel = self.kernel, iterations=self.iterations)
        if self.showOutput:
            cv2.imshow(self.outputWindowName, output)
        return output
class Closing(PostProcessingBlock):
    def __init__(self,kernel,iterations, showOutput = False, outputWindowName = 'Closing') -> None:
        self.showOutput = showOutput
        self.outputWindowName = outputWindowName
        self.kernel = kernel
        self.iterations = iterations
        

    def run(self, input):
        output = input.copy()
        cv2.dilate(output,kernel = self.kernel, iterations=self.iterations)
        if self.showOutput:
            cv2.imshow((self.outputWindowName+'2'), output)
        
        cv2.erode(output,kernel = self.kernel, iterations=self.iterations)
        if self.showOutput:
            cv2.imshow(self.outputWindowName, output)
        return output
class Opening(PostProcessingBlock):
    def __init__(self,kernel,iterations, showOutput = True, outputWindowName = 'Opening') -> None:
        self.showOutput = showOutput
        self.outputWindowName = outputWindowName
        self.kernel = kernel
        self.iterations = iterations
        

    def run(self, input):
        output = input.copy()
        cv2.erode(output,kernel = self.kernel, iterations=self.iterations)
        if self.showOutput:
            cv2.imshow((self.outputWindowName+'2'), output)
        
        cv2.dilate(output,kernel = self.kernel, iterations=self.iterations)
        if self.showOutput:
            cv2.imshow(self.outputWindowName, output)
        return output

pipeline_QR = PostProcessingPipeline([
    ConvertBGR2Gray(),
    AverageBlur(filterSize=1),
    Threshold(220,255)
    ])
def GaussianBlur_QR(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_QR.blocks[1] = AverageBlur(GaussianBlur_V_R)

pipeline_4 = PostProcessingPipeline([
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=False),
    ConvertBGR2HSV(),
    HSVThreshold2(lowerBound = np.array([0,34,38]),upperBound=np.array([99,255,254]),trackbar= True,WindName='trackbar_4'),
    ConvertHSV2BGR(),
    ConvertBGR2Gray(showOutput=True, outputWindowName='trackbar_4'),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(5656, 100000,namedWindow=False,trackbar= True,WindName='trackbar_4'),
    DetectShapes2(epsilon= 0.1,trackbar= True,WindName='trackbar_4')
    ])
def GaussianBlur_4(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_4.blocks[0] = AverageBlur(filterSize=GaussianBlur_V_R)

pipeline_1 = PostProcessingPipeline([
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=False),
    ConvertBGR2HSV(),
    HSVThreshold2(lowerBound = np.array([0,34,38]),upperBound=np.array([99,255,254]),trackbar= True,WindName='trackbar_1'),
    ConvertHSV2BGR(),
    ConvertBGR2Gray(showOutput=True, outputWindowName='trackbar_1'),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(5656, 100000,namedWindow=False,trackbar= True,WindName='trackbar_1'),
    DetectShapes2(epsilon= 0.1,trackbar= True,WindName='trackbar_1')
    ])
def GaussianBlur_1(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_1.blocks[0] = AverageBlur(GaussianBlur_V_R)    

pipeline_2 = PostProcessingPipeline([
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=False),
    ConvertBGR2HSV(),
    HSVThreshold2(lowerBound = np.array([0,34,38]),upperBound=np.array([99,255,254]),trackbar= True,WindName='trackbar_2'),
    ConvertHSV2BGR(),
    ConvertBGR2Gray(showOutput=True, outputWindowName='trackbar_2'),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(5656, 100000,namedWindow=False,trackbar= True,WindName='trackbar_2'),
    DetectShapes2(epsilon= 0.1,trackbar= True,WindName='trackbar_2')
    ])
def GaussianBlur_2(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_2.blocks[0] = AverageBlur(GaussianBlur_V_R)    

pipeline_3 = PostProcessingPipeline([
    AverageBlur(filterSize=10),
    IntensityPower(power=1.5,showOutput=False),
    ConvertBGR2HSV(),
    HSVThreshold2(lowerBound = np.array([0,34,38]),upperBound=np.array([99,255,254]),trackbar= True,WindName='trackbar_3'),
    ConvertHSV2BGR(),
    ConvertBGR2Gray(showOutput=True, outputWindowName='trackbar_3'),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(5656, 100000,namedWindow=False,trackbar= True,WindName='trackbar_3'),
    DetectShapes2(epsilon= 0.1,trackbar= True,WindName='trackbar_3')
    ])
def GaussianBlur_3(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_3.blocks[0] = AverageBlur(GaussianBlur_V_R)    

cv2.createTrackbar("Gaussian", "trackbar_1", pipeline_1.blocks[0].filterSize, 125, GaussianBlur_1)
cv2.createTrackbar("Gaussian", "trackbar_2", pipeline_2.blocks[0].filterSize, 125, GaussianBlur_2)
cv2.createTrackbar("Gaussian", "trackbar_3", pipeline_3.blocks[0].filterSize, 125, GaussianBlur_3)
cv2.createTrackbar("Gaussian", "trackbar_4", pipeline_4.blocks[0].filterSize, 125, GaussianBlur_4)

def PipeRes(frame):
    pipeRes = [0,0,0,0]
    Firkan_Liste = []
    pipeRes[0] = pipeline_1.run(frame)
    pipeRes[1] = pipeline_2.run(frame)
    pipeRes[2] = pipeline_3.run(frame)
    pipeRes[3] = pipeline_4.run(frame)
    m1 = 1
    for m in pipeRes:
        shapeImg = frame.copy()
        for s in m:
            print(s)
            cv2.drawContours(shapeImg, [s.contour], -1, (0, 0, 150), 1)
            if s.points == 4:
                cv2.drawContours(shapeImg, [s.contour], -1, (0, 255, 0), 4)
                Firkan_Liste.append([m1,s])
                break
        
        m1 +=1
    return Firkan_Liste
