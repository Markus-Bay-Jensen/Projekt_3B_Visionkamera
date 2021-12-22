import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
        
class Threshold2(Threshold):
    def __init__(self, lowerBound: int, upperBound: int, mode: int = cv2.THRESH_BINARY, showOutput: bool = False, outputWindowName: str = 'Threshold',trackbar: bool = False,namedWindow=True,WindName='HSV Treshold Trackbars') -> None:
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
    def __init__(self, minArea: float, maxArea: float, printDebug: bool = False,trackbar: bool = False,namedWindow=True,WindName='HSV Treshold Trackbars') -> None:
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
        cv2.createTrackbar("Conto_minR", self.WindName, self.minArea, 100000, self.Conto_minR)
        cv2.createTrackbar("Conto_maxR", self.WindName, self.maxArea, 100000, self.Conto_maxR)

    def Conto_minR(self, value: int):
        if value < 0 or value > 100000:
            return
        self.minArea = value

    def Conto_maxR(self, value: int):
        if value < 0 or value > 100000:
            return
        self.maxArea = value

class HSVThreshold2(HSVThreshold):
    def __init__(self, lowerBound: np.array = np.array([0, 0, 0]), upperBound: np.array = np.array([255, 255, 255]), trackbar: bool = False, showOutput: bool = False, outputWindowName: str = 'HSVThreshold',namedWindow=True,WindName='HSV Treshold Trackbars') -> None:
        self.showOutput = showOutput
        self.outputWindowName = outputWindowName
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.WindName = WindName
        if trackbar:
            self.createTrackbar()
        self.namedWindow = namedWindow

    def createTrackbar(self):
        #if self.namedWindow:
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
    def __init__(self,kernel,iterations, showOutput = True, outputWindowName = 'Closing') -> None:
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

#cv2.namedWindow("Trackbars")
cap = OAKCamColorDepth(640,400)
def GaussianBlur_4(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_4.blocks[1] = AverageBlur(GaussianBlur_V_R)
def GaussianBlur_1(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_1.blocks[0] = AverageBlur(GaussianBlur_V_R)

'''

pipeline_4 = PostProcessingPipeline([
    GetGreenChannel(),
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=True),
    Threshold2(65, 255, showOutput = True,outputWindowName='Threshold 4',namedWindow=True,WindName='Trackbars',trackbar=True),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(1494, 3372,namedWindow=False,WindName='Trackbars',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])'''
#cv2.createTrackbar("Gaussian_4", "Trackbars", 1, 125, GaussianBlur_4) 
pipeline_1 = PostProcessingPipeline([
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=True),
    ConvertBGR2HSV(),
    HSVThreshold2(trackbar=True,WindName='Trackbars',namedWindow=True),
    ConvertHSV2BGR(showOutput=True),
    GetGreenChannel(),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(1494, 3372,namedWindow=False,WindName='Trackbars',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
cv2.createTrackbar("Gaussian_1", "Trackbars", 1, 125, GaussianBlur_1)





cv2.namedWindow('Trackbars')
while(True):
    Firkan_Liste = []
    
    frame = cap.getPreviewFrame()
    
    cv2.imshow("frame", frame)

    pipeRes_R = pipeline_1.run(frame)
    print(pipeRes_R)
    
    shapeImg_R = frame.copy()
    
    for s in pipeRes_R :
        
        cv2.drawContours(shapeImg_R, [s.contour], -1, (0, 0, 150), 1)
        
        if s.points == 4:
            cv2.drawContours(shapeImg_R, [s.contour], -1, (0, 255, 0), 4)
            Firkan_Liste.append([s.center,s.contour])

    cv2.imshow("Shapes R", shapeImg_R)
    print(Firkan_Liste)
    
    key = cv2.waitKey(1000)

    if key == ord('q'):
        break

    elif key == ord('t'):
        cap.triggerAutoFocus()

    elif key == ord('a'):
        cap.startContinousAutoFocus()


cv2.destroyAllWindows()
cv2.destroyAllWindows()