
import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *

cap = OAKCamColorDepth()


cv2.namedWindow("Trackbars")

pipelineDepthFrame = PostProcessingPipeline([
    GaussianBlur(201),
    Threshold(75, 100,outputWindowName='Threshold Frame Depth'),
    DetectContours(drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours(800, 8000, printDebug = True),
    DetectShapes(printResult = True)
])
thresholdmin_Val = 0
thresholdContours_Val = 0
def gaussianBlur(newVal):
    if newVal < 1:
        return
    if newVal % 2 == 0:
        newVal = newVal - 1
    print(newVal)
    pipelineDepthFrame.blocks[0] = GaussianBlur(newVal, True)
cv2.createTrackbar("GaussianBlur", "Trackbars", 1, 99, gaussianBlur)
def thresholdmin(newVal2):
    print(newVal2)
    thresholdmin_Val = newVal2
cv2.createTrackbar("Threshold min", "Trackbars", 1, 255, thresholdmin)
def thresholdmax(newVal3):
    print(newVal3)
    pipelineDepthFrame.blocks[1] = Threshold(thresholdmin_Val, newVal3, showOutput = True,outputWindowName='Threshold Frame Depth')
cv2.createTrackbar("Threshold max", "Trackbars", 1, 255, thresholdmax)
def thresholdContoursmin(newVal4):
    print(newVal4)
    thresholdContours_Val = newVal4
cv2.createTrackbar("Contours min", "Trackbars", 1, 30000, thresholdContoursmin)
def thresholdContoursmax(newVal5):
    print(newVal5)
    pipelineDepthFrame.blocks[3] = ThresholdContours(thresholdContours_Val, newVal5, printDebug = True)
cv2.createTrackbar("Contours max", "Trackbars", 1, 30000, thresholdContoursmax)

pipelineFrame = PostProcessingPipeline([
    #GetRedChannel(),
    ConvertToGray(showOutput = True, outputWindowName = 'test window Frame2'),
    #AverageBlur(7, True),
    GaussianBlur(1001),
    #BilateralFilter(5, 10, 10, True)
    #LaplacianSharpen(0.5, True, True)
    #UnsharpMasking(7, True)
    Threshold(175, 255, outputWindowName='Threshold Frame2'),
    #AdaptiveThreshold(200, 11, 2, showOutput = True),
    #OtsuBinarization(True)
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    #ThresholdContours(80, 10000, printDebug = True),
    DetectShapes(printResult = True)
])
while(True):


    frameDepthFrame = cap.getDepthFrame()
    frameFrame = cap.getPreviewFrame()
    

    pipeRes = pipelineDepthFrame.run(frameDepthFrame)
    pipeRes2 = pipelineFrame.run(frameFrame)
    '''
    shapeImg = frameDepthFrame.copy()
    for s in pipeRes:
        cv2.drawContours(shapeImg, [s.contour], -1, (0, 255, 0), 2)
        cv2.putText(shapeImg, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

    shapeImg2 = frameFrame.copy()
    for s in pipeRes2:
        cv2.drawContours(shapeImg2, [s.contour], -1, (0, 0, 255), 2)
        cv2.putText(shapeImg2, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
    
    shapeImg3 = frameFrame.copy()
    for s in pipeRes:
        cv2.drawContours(shapeImg2, [s.contour], -1, (255, 0, 0), 2)
        cv2.putText(shapeImg2, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
        cv2.drawContours(shapeImg, [s.contour], -1, (255, 0, 0), 2)
        cv2.putText(shapeImg, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
    '''
    cv2.imshow("Webcam", frameDepthFrame)
    #cv2.imshow("Shapes", shapeImg)
    cv2.imshow("Webcam2", frameFrame)
    #cv2.imshow("Shapes2", shapeImg2)
    #cv2.imshow("Shapes3", shapeImg3)
    if cv2.waitKey(50) == ord('q'):
        break


cv2.destroyAllWindows()
