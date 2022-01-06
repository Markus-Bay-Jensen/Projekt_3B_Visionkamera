import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
import Internetkommunikation




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
cv2.namedWindow('Trackbar QR2')
cv2.namedWindow('Trackbar 1')
cv2.namedWindow('Trackbar 2')
cv2.namedWindow('Trackbar 3')
cv2.namedWindow('Trackbar 4')
pipeline_QR = PostProcessingPipeline([
    ConvertBGR2Gray(),
    AverageBlur(filterSize=1),
    #LaplacianSharpen()
    #UnsharpMasking()
    #Sobel4()
    Threshold(220,255)
    ])
def GaussianBlur_QR(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_QR.blocks[1] = AverageBlur(GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_QR", "Trackbar QR2", 1, 125, GaussianBlur_QR)
pipeline_4 = PostProcessingPipeline([
    AverageBlur(filterSize=5),
    IntensityPower(power=1.5,showOutput=False),
    ConvertBGR2HSV(),
    HSVThreshold2(lowerBound = np.array([0,34,38]),upperBound=np.array([99,255,254]),trackbar=True,WindName='Trackbar 4',namedWindow=True),
    ConvertHSV2BGR(showOutput=False),
    GetGreenChannel(),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(5656, 100000,namedWindow=False,WindName='Trackbar 4',trackbar=False),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_4(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_4.blocks[0] = AverageBlur(filterSize=GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_4", "Trackbar 4", 5, 125, GaussianBlur_4) 
pipeline_1 = PostProcessingPipeline([
    AverageBlur(filterSize=23),
    IntensityPower(power=1.5,showOutput=False),
    ConvertBGR2HSV(),
    HSVThreshold2(lowerBound = np.array([0,229,159]),upperBound=np.array([87,243,255]),trackbar=True,WindName='Trackbar 1',namedWindow=True),
    ConvertHSV2BGR(showOutput=False),
    GetGreenChannel(),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(10007, 18637,namedWindow=False,WindName='Trackbar 1',trackbar=False),
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
    AverageBlur(filterSize=15),
    IntensityPower(power=1.5,showOutput=False),
    ConvertBGR2HSV(),
    HSVThreshold2(lowerBound = np.array([89,65,201]),upperBound=np.array([154,255,223]),trackbar=True,WindName='Trackbar 2',namedWindow=True),
    ConvertHSV2BGR(showOutput=False),
    GetGreenChannel(),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(18419, 28934,namedWindow=False,WindName='Trackbar 2',trackbar=False),
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
    IntensityPower(power=1.5,showOutput=False),
    ConvertBGR2HSV(),
    HSVThreshold2(lowerBound = np.array([64,35,103]),upperBound=np.array([87,230,140]),trackbar=True,WindName='Trackbar 3',namedWindow=True),
    ConvertHSV2BGR(showOutput=False),
    GetGreenChannel(),
    Closing((5,5),5),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(8920, 16896,namedWindow=False,WindName='Trackbar 3',trackbar=False),
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
                break
        cv2.imshow("Shapes "+str(m1), shapeImg)
        m1 +=1
    return Firkan_Liste



TCP = Internetkommunikation.TCP_pi_Server(HOST='',PORT=23)
F1 = 0
F2 = 0
F3 = 0
Robot_o = QR.Omregning('QR.txt')
cap = OAKCamColor(1080,1080)
qr = True
qr2 = False
Firkan = False
tcp_IP = False

if tcp_IP:
    TCP.TCP_Aben()

while(True):
    frame = cap.getPreviewFrame()
    frame = Robot_o.Rotering(frame)
    #print('')

    if tcp_IP:
            Besked = TCP.TCP_Modtaget()
            if Besked == 'q':
                break
            elif Besked == 't':
                cap.triggerAutoFocus()
            elif Besked == 'a':
                cap.startContinousAutoFocus()

    if qr:
        
        
            

        while(True):
            
            frame_QR = cap.getPreviewFrame()
            frame_QR = pipeline_QR.run(frame_QR.copy())
            frame_QR,Break= Robot_o.Nulstilling(frame_QR)
            cv2.imshow("QR", frame_QR)
            key = cv2.waitKey(500)
            if Break:
                break
        #break
        qr = False
        #Firkan = True

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
        Firkan_Liste = PipeRes(frame.copy())
        frame_Firkan =frame.copy()
        for f in Firkan_Liste:
            XY , V , frame_Firkan = Robot_o.Omregning_V(f[2],frame_Firkan)
            print('CM:',XY,' P:',f[1],'V:',V,' F:',f[0])
            Firkan_cm.append([XY,V,f[0]]) 
        cv2.imshow("Firkan", frame_Firkan)
        frame = frame_Firkan

    if tcp_IP:
        M2 = [-0.23339,0.02856,0.2481,0,3.0765,0,4]
        M = '['+str(M2[0])+','+str(M2[1])+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+',4]'
        F1_ = False
        F2_ = False
        F3_ = False
        '''
        for F_cm in Firkan_cm:
            if F_cm[2] == 1:
                F1_ = True
            if F_cm[2] == 2:
                F2_ = True
            if F_cm[2] == 3:
                F3_ = True
        F1_3 = F1_ and ((F2_==False or F1 <= F2) and (F3_==False or F1 <= F3))
        F2_3 = F2_ and ((F1_==False or F2 < F1) and (F3_==False or F2 <= F3))
        F3_3 = F3_ and ((F1_==False or F3 < F1) and (F2_==False or F3 < F2))
        for F_cm in Firkan_cm:
            if F1_3 and F_cm[2] == 1:
                M  = str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+str(F1_3[2])
                break
            if F2_3 and F_cm[2] == 2:
                M  = str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+str(F1_3[2])
                break
            if F3_3 and F_cm[2] == 3:
                M  = str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+str(F1_3[2])
                break
            if F_cm[2] >= 4 and F1_ == False and F2_ == False and F3_ == False:
                M  = str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+str(F1_3[2])
                break'''
        TCP.TCP_Send(M)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(2000)
    if k == ord('q'):
        break
    elif k == ord('t'):
        cap.triggerAutoFocus()
    elif k == ord('a'):
        cap.startContinousAutoFocus()

#TCP.TCP_Luk()
#TCP.TCP_Luk_Luk()
cv2.destroyAllWindows()
cv2.destroyAllWindows()