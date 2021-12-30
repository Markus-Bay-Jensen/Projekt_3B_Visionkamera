import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
#import AI 
#ai = AI.AI()
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
cv2.namedWindow("Trackbars")
cap = OAKCamColorDepth(640,400)

pipeline_4 = PostProcessingPipeline([
    GetGreenChannel(),
    GaussianBlur(6),
    
    IntensityPower(power=1.5,showOutput=True),
    Threshold(65, 255, showOutput = True,outputWindowName='Threshold R'),

    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours(1494, 3372),
    DetectShapes(epsilon= 0.1)   
])
GaussianBlur_V_R = 0
def GaussianBlur_R(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_R.blocks[1] = AverageBlur(GaussianBlur_V_R)


def Threshold_R(newVal):
    Threshold_min_V_R = newVal
    Threshold_max_V_R = pipeline_R.blocks[3].upperBound
    pipeline_R.blocks[3] = Threshold(Threshold_min_V_R,Threshold_max_V_R, showOutput = True, outputWindowName='Threshold R')
    print(Threshold_min_V_R,'-',Threshold_max_V_R)

def ThresholdContours_min_R(newVal):
    min_V = newVal
    max_V = pipeline_R.blocks[7].maxArea
    pipeline_R.blocks[7] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)
def ThresholdContours_max_R(newVal):
    min_V = pipeline_R.blocks[7].minArea
    max_V = newVal
    pipeline_R.blocks[7] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)

cv2.createTrackbar("Gaussian_R", "Trackbars", 1, 125, GaussianBlur_R)

cv2.createTrackbar("Thres_R", "Trackbars", 1, 255, Threshold_R)
cv2.createTrackbar("Conto_minR", "Trackbars", 1, 10000, ThresholdContours_min_R)
cv2.createTrackbar("Conto_maxR", "Trackbars", 1, 10000, ThresholdContours_max_R)

while(True):
    Firkan_Liste = []
    frameD = cap.getDepthFrame()
    frame = cap.getPreviewFrame()

    #QRframe , m= QR.QR(frame.copy())
    #print(QRframe)
    cv2.imshow("frameD", frameD)
    cv2.imshow("frame", frame)

    pipeRes_R = pipeline_R.run(frame)
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