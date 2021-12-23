import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
import Internetkommunikation

TCP = Internetkommunikation.TCP_pi_Server(HOST='',PORT=5120)


class Threshold2(Threshold):
    def __init__(self, lowerBound: int, upperBound: int, mode: int = cv2.THRESH_BINARY, showOutput: bool = False, outputWindowName: str = 'Threshold',trackbar: bool = False,namedWindow=True,WindName='Threshold') -> None:
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
    def __init__(self, minArea: float, maxArea: float, printDebug: bool = False,trackbar: bool = False,namedWindow=True,WindName='ThresholdContours') -> None:
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
cv2.namedWindow('Trackbar QR')
cv2.namedWindow('Trackbar 1')
cv2.namedWindow('Trackbar 2')
cv2.namedWindow('Trackbar 3')
cv2.namedWindow('Trackbar 4')
pipeline_QR = PostProcessingPipeline([
    GetGreenChannel(),
    AverageBlur(filterSize=2)
    ])
def GaussianBlur_QR(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_QR.blocks[0] = AverageBlur(GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_QR", "Trackbar QR", 1, 125, GaussianBlur_QR)
pipeline_4 = PostProcessingPipeline([
    GetGreenChannel(),
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=True),
    Threshold2(65, 255, showOutput = True,outputWindowName='Threshold 4',namedWindow=True,WindName='Trackbar 4',trackbar=True),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(1494, 3372,namedWindow=False,WindName='Trackbar 4',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_4(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_4.blocks[1] = AverageBlur(filterSize=GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_4", "Trackbar 4", 5, 125, GaussianBlur_4) 
pipeline_1 = PostProcessingPipeline([
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=True),
    ConvertBGR2HSV(),
    HSVThreshold2(trackbar=True,WindName='Trackbar 1',namedWindow=True),
    ConvertHSV2BGR(showOutput=True),
    GetGreenChannel(),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(1494, 3372,namedWindow=False,WindName='Trackbar 1',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_1(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_1.blocks[0] = AverageBlur(GaussianBlur_V_R)    
cv2.createTrackbar("Gaussian_1", "Trackbar 1", 5, 125, GaussianBlur_1)
pipeline_2 = PostProcessingPipeline([
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=True),
    ConvertBGR2HSV(),
    HSVThreshold2(trackbar=True,WindName='Trackbar 2',namedWindow=True),
    ConvertHSV2BGR(showOutput=True),
    GetGreenChannel(),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(1494, 3372,namedWindow=False,WindName='Trackbar 2',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_2(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_2.blocks[0] = AverageBlur(GaussianBlur_V_R)    
cv2.createTrackbar("Gaussian_2", "Trackbar 2", 5, 125, GaussianBlur_2)
pipeline_3 = PostProcessingPipeline([
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=True),
    ConvertBGR2HSV(),
    HSVThreshold2(trackbar=True,WindName='Trackbar 3',namedWindow=True),
    ConvertHSV2BGR(showOutput=True),
    GetGreenChannel(),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(1494, 3372,namedWindow=False,WindName='Trackbar 3',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_3(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_3.blocks[0] = AverageBlur(GaussianBlur_V_R)    
cv2.createTrackbar("Gaussian_3", "Trackbar 3", 5, 125, GaussianBlur_3)

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
            cv2.drawContours(shapeImg, [s.contour], -1, (0, 0, 150), 1)
            if s.points == 4:
                cv2.drawContours(shapeImg, [s.contour], -1, (0, 255, 0), 4)
                Firkan_Liste.append([m1,s.center,s.contour])
        cv2.imshow("Shapes "+str(m1), shapeImg)
        m1 +=1
    return Firkan_Liste
Robot_o = QR.Omregning('QR.txt')
cap = OAKCamColorDepth(900,800)
qr = True
qr2 = True
Firkan = False
tcp_PI = False
while(True):
    frame = cap.getPreviewFrame()
    
    if qr:
        
        while(True):
            
            frame = cap.getPreviewFrame()
            #pipeline_QR.run(frame.copy())
            Data_QR,Robot = QR.QR(frame)
            cv2.imshow("QR", frame)
            key = cv2.waitKey(100)
            m2 = 0
            for m in Robot:
                m2+=1
                if m2 >=3:
                    break
            if m2 >=3:
                break
        Robot_o.Nulstilling(Robot)
        qr = False

    if qr2:
        Data_QR,Robot = QR.QR(frame)
        for mm in Data_QR:
            XY10 = Robot_o.Omregning(mm[1])
            cv2.putText(frame,str(XY10),(mm[1][0],mm[1][1]+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2 )
            cv2.putText(frame,str(mm[1]),(mm[1][0],mm[1][1]+40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2 )
        for mm in Robot:
            XY10 = Robot_o.Omregning(mm[1])
            cv2.putText(frame,str(XY10),(mm[1][0],mm[1][1]+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2 )
            cv2.putText(frame,str(mm[1]),(mm[1][0],mm[1][1]+40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2 )
            

    if Firkan:
        Firkan_cm = []
        frame = cap.getPreviewFrame()
        Firkan_Liste = PipeRes(frame.copy())
        for f in Firkan_Liste:
            XY = Robot_o.Omregning(f[1])
            print('CM:',XY,' P:',f[1],' F:',f[0])
            Firkan_cm.append(XY,f[0]) 

    if tcp_PI:
        TCP.TCP_Aben()
        Besked = TCP.TCP_Modtaget()
        TCP.TCP_Send('0')

    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    elif k == ord('t'):
        cap.triggerAutoFocus()
    elif k == ord('a'):
        cap.startContinousAutoFocus()


cv2.destroyAllWindows()
cv2.destroyAllWindows()