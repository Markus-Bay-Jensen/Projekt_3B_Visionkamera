import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
import Internetkommunikation

while True:
    ny = False
    file = QR.file(input("Indtast vision kamera navn: ")+'.txt')
    M = input('er det et nyt kamera (y/n) ')
    if M == 'n':
        file_data = file.Aben()
        break
    elif M == 'y':
        M = input('Vil du låne filer fra et andet kamera (y/n) ')
        if M == 'y':
            file_data = file.Aben(input("Indtast vision kamera navn: ")+'.txt')
            file.Gem()
            break
        elif M == 'n':
            file_data = {'filterSize_Q':1,'filterSize_1':1,'filterSize_2':1,'filterSize_3':1,'filterSize_4':1,'Threshold_1':1,'ThresholdContours_min_1':1,'ThresholdContours_max_1':1,'Threshold_2':1,'ThresholdContours_min_2':1,'ThresholdContours_max_2':1,'Threshold_3':1,'ThresholdContours_min_3':1,'ThresholdContours_max_3':1,'Threshold_4':1,'ThresholdContours_min_4':1,'ThresholdContours_max_4':1,'lowerBound_1':'0,0,0','upperBound_1':'0,0,0','lowerBound_2':'0,0,0','upperBound_2':'0,0,0','lowerBound_3':'0,0,0','upperBound_3':'0,0,0'}
            file_data['PORT'] = input('PORT :')
            file_data['HOST'] = input('HOST :')
            M = input('Har du en fil med QR-kode positioner (y/n) ')
            file_data['file_QR'] = input('file_QR :')+'.txt'
            if M =='n':
                Q = open(file_data['file_QR'], "w")
                while True: 
                    print('Når du er færdig med at skrive QR-kode er en test q \n Første tal er QR-koden nummer \n Næste tal er QR-koden X aksen \n Sidste nummer er QR-koden Y X aksen \n Den skal se sådan her ud 0,0,0')
                    M = input('>>>>>>>>')
                    if M =='q': break
                    Q.write(M+',\n')
                Q.close()
            file.Gem(file_data)
            break


TCP = Internetkommunikation.TCP_pi_Server(HOST=file_data['HOST'],PORT=int(file_data['PORT']))



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
    AverageBlur(filterSize=int(file_data['filterSize_Q']))
    ])
def GaussianBlur_QR(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_QR.blocks[0] = AverageBlur(GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_QR", "Trackbar QR", int(file_data['filterSize_Q']), 125, GaussianBlur_QR)

nr = '4'
pipeline_4 = PostProcessingPipeline([
    GetGreenChannel(),
    AverageBlur(filterSize=int(file_data['filterSize_'+nr])),
    IntensityPower(power=1.5,showOutput=True),
    Threshold2(int(file_data['Threshold_'+nr]), 255, showOutput = True,outputWindowName='Threshold 4',namedWindow=True,WindName='Trackbar 4',trackbar=True),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(int(file_data['ThresholdContours_min_'+nr]),int(file_data['ThresholdContours_max_'+nr]) ,namedWindow=False,WindName='Trackbar 4',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_4(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    file_data['filterSize_'+nr] = GaussianBlur_V_R
    pipeline_4.blocks[1] = AverageBlur(filterSize=GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_"+nr, "Trackbar "+nr, int(file_data['filterSize_'+nr]), 125, GaussianBlur_4) 

nr = '3'
lowerBound = file_data['lowerBound_'+nr]
lowerBound = lowerBound.split(",")
lowerBound = [int(lowerBound[0]),int(lowerBound[1]),int(lowerBound[2])]
lowerBound = np.array(lowerBound)
upperBound = file_data['upperBound_'+nr]
upperBound = upperBound.split(",")
upperBound = [int(upperBound[0]),int(upperBound[1]),int(upperBound[2])]
upperBound = np.array(upperBound)
pipeline_3 = PostProcessingPipeline([
    AverageBlur(filterSize=int(file_data['filterSize_'+nr])),
    ConvertBGR2HSV(),
    HSVThreshold2(trackbar=True,WindName='Trackbar 1',namedWindow=True,lowerBound=lowerBound,upperBound=upperBound),
    ConvertHSV2BGR(showOutput=True),
    IntensityPower(power=1.5,showOutput=True),
    Threshold2(int(file_data['Threshold_'+nr]), 255, showOutput = True,outputWindowName='Threshold 4',namedWindow=True,WindName='Trackbar 4',trackbar=True),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(int(file_data['ThresholdContours_min_'+nr]),int(file_data['ThresholdContours_max_'+nr]) ,namedWindow=False,WindName='Trackbar 4',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_3(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    file_data['filterSize_'+nr] = GaussianBlur_V_R
    pipeline_4.blocks[0] = AverageBlur(filterSize=GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_"+nr, "Trackbar "+nr, int(file_data['filterSize_'+nr]), 125, GaussianBlur_3) 

nr = '2'
lowerBound = file_data['lowerBound_'+nr]
lowerBound = lowerBound.split(",")
lowerBound = [int(lowerBound[0]),int(lowerBound[1]),int(lowerBound[2])]
lowerBound = np.array(lowerBound)
upperBound = file_data['upperBound_'+nr]
upperBound = upperBound.split(",")
upperBound = [int(upperBound[0]),int(upperBound[1]),int(upperBound[2])]
upperBound = np.array(upperBound)
pipeline_2 = PostProcessingPipeline([
    AverageBlur(filterSize=int(file_data['filterSize_'+nr])),
    ConvertBGR2HSV(),
    HSVThreshold2(trackbar=True,WindName='Trackbar 1',namedWindow=True,lowerBound=lowerBound,upperBound=upperBound),
    ConvertHSV2BGR(showOutput=True),
    IntensityPower(power=1.5,showOutput=True),
    Threshold2(int(file_data['Threshold_'+nr]), 255, showOutput = True,outputWindowName='Threshold 4',namedWindow=True,WindName='Trackbar 4',trackbar=True),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(int(file_data['ThresholdContours_min_'+nr]),int(file_data['ThresholdContours_max_'+nr]) ,namedWindow=False,WindName='Trackbar 4',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_2(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    file_data['filterSize_'+nr] = GaussianBlur_V_R
    pipeline_4.blocks[0] = AverageBlur(filterSize=GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_"+nr, "Trackbar "+nr, int(file_data['filterSize_'+nr]), 125, GaussianBlur_2) 

nr = '1'
lowerBound = file_data['lowerBound_'+nr]
lowerBound = lowerBound.split(",")
lowerBound = [int(lowerBound[0]),int(lowerBound[1]),int(lowerBound[2])]
lowerBound = np.array(lowerBound)
upperBound = file_data['upperBound_'+nr]
upperBound = upperBound.split(",")
upperBound = [int(upperBound[0]),int(upperBound[1]),int(upperBound[2])]
upperBound = np.array(upperBound)
pipeline_1 = PostProcessingPipeline([
    AverageBlur(filterSize=int(file_data['filterSize_'+nr])),
    ConvertBGR2HSV(),
    HSVThreshold2(trackbar=True,WindName='Trackbar 1',namedWindow=True,lowerBound=lowerBound,upperBound=upperBound),
    ConvertHSV2BGR(showOutput=True),
    IntensityPower(power=1.5,showOutput=True),
    Threshold2(int(file_data['Threshold_'+nr]), 255, showOutput = True,outputWindowName='Threshold 4',namedWindow=True,WindName='Trackbar 4',trackbar=True),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours2(int(file_data['ThresholdContours_min_'+nr]),int(file_data['ThresholdContours_max_'+nr]) ,namedWindow=False,WindName='Trackbar 4',trackbar=True),
    DetectShapes(epsilon= 0.1)
    ])
def GaussianBlur_1(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    file_data['filterSize_'+nr] = GaussianBlur_V_R
    pipeline_4.blocks[0] = AverageBlur(filterSize=GaussianBlur_V_R)
cv2.createTrackbar("Gaussian_"+nr, "Trackbar "+nr, int(file_data['filterSize_'+nr]), 125, GaussianBlur_1) 

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







Robot_o = QR.Omregning(file_data['file_QR'])
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
            XY = Robot_o.Omregning_V(f[1])
            print('CM:',XY,' P:',f[1],' F:',f[0])
            Firkan_cm.append(XY,V,f[0]) 

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