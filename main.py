import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
cv2.namedWindow("Trackbars")
cap = OAKCamColorDepth(640,400)

pipeline_D = PostProcessingPipeline([
    GaussianBlur(1),
    Threshold(1, 255, showOutput = True, outputWindowName='Threshold D'),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours(100, 8000, printDebug = True),
    DetectShapes(printResult = True)    
    
])
GaussianBlur_V_D = 0
def GaussianBlur_D(newVal):
    if newVal < 1:
        GaussianBlur_V_D = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_D = newVal - 1
    pipeline_D.blocks[0] = GaussianBlur(GaussianBlur_V_D)

def Threshold_min_D(newVal):
    Threshold_min_V = newVal
    Threshold_max_V = pipeline_D.blocks[1].upperBound
    pipeline_D.blocks[1] = Threshold(Threshold_min_V,Threshold_max_V, showOutput = True, outputWindowName='Threshold D')
    print(Threshold_min_V,'-',Threshold_max_V)
def Threshold_max_D(newVal):
    Threshold_min_V = pipeline_D.blocks[1].lowerBound
    Threshold_max_V = newVal
    pipeline_D.blocks[1] = Threshold(Threshold_min_V,Threshold_max_V, showOutput = True, outputWindowName='Threshold D')
    print(Threshold_min_V,'-',Threshold_max_V)

def ThresholdContours_min_D(newVal):
    min_V = newVal
    max_V = pipeline_D.blocks[3].maxArea
    pipeline_D.blocks[3] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)
def ThresholdContours_max_D(newVal):
    min_V = pipeline_D.blocks[3].minArea
    max_V = newVal
    pipeline_D.blocks[3] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)


pipeline_R = PostProcessingPipeline([
    GetRedChannel(),
    GaussianBlur(1),
    Threshold(1, 255, showOutput = True,outputWindowName='Threshold R'),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours(100, 8000, printDebug = True),
    DetectShapes(printResult = True)   
])
GaussianBlur_V_R = 0
def GaussianBlur_R(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_R.blocks[1] = AverageBlur(GaussianBlur_V_R)
def Threshold_min_R(newVal):
    Threshold_min_V_R = newVal
    Threshold_max_V_R = pipeline_R.blocks[2].upperBound
    pipeline_R.blocks[2] = Threshold(Threshold_min_V_R,Threshold_max_V_R, showOutput = True, outputWindowName='Threshold R')
    print(Threshold_min_V_R,'-',Threshold_max_V_R)
def Threshold_max_R(newVal):
    Threshold_min_V_R = pipeline_R.blocks[2].lowerBound
    Threshold_max_V_R = newVal
    pipeline_R.blocks[2] = Threshold(Threshold_min_V_R,Threshold_max_V_R, showOutput = True, outputWindowName='Threshold R')
    print(Threshold_min_V_R,'-',Threshold_max_V_R)

def ThresholdContours_min_R(newVal):
    min_V = newVal
    max_V = pipeline_R.blocks[4].maxArea
    pipeline_R.blocks[4] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)
def ThresholdContours_max_R(newVal):
    min_V = pipeline_R.blocks[4].minArea
    max_V = newVal
    pipeline_R.blocks[4] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)


pipeline_G = PostProcessingPipeline([
    GetGreenChannel(),
    GaussianBlur(1),
    Threshold(1, 255, showOutput = True,outputWindowName='Threshold G'),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours(100, 8000, printDebug = True),
    DetectShapes(printResult = True)    
])
GaussianBlur_V_G = 0
def GaussianBlur_G(newVal):
    if newVal < 1:
        GaussianBlur_V_G = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_G = newVal - 1
    pipeline_G.blocks[1] = AverageBlur(GaussianBlur_V_G)
def Threshold_min_G(newVal):
    Threshold_min_V = newVal
    Threshold_max_V = pipeline_G.blocks[2].upperBound
    pipeline_G.blocks[2] = Threshold(Threshold_min_V,Threshold_max_V, showOutput = True, outputWindowName='Threshold G')
    print(Threshold_min_V,'-',Threshold_max_V)
def Threshold_max_G(newVal):
    Threshold_min_V = pipeline_G.blocks[2].lowerBound
    Threshold_max_V = newVal
    pipeline_G.blocks[2] = Threshold(Threshold_min_V,Threshold_max_V, showOutput = True, outputWindowName='Threshold G')
    print(Threshold_min_V,'-',Threshold_max_V)

def ThresholdContours_min_G(newVal):
    min_V = newVal
    max_V = pipeline_G.blocks[4].maxArea
    pipeline_G.blocks[4] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)
def ThresholdContours_max_G(newVal):
    min_V = pipeline_G.blocks[4].minArea
    max_V = newVal
    pipeline_G.blocks[4] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)


pipeline_B = PostProcessingPipeline([
    GetBlueChannel(),
    GaussianBlur(1),
    Threshold(1, 255, showOutput = True,outputWindowName='Threshold B'),
    DetectContours( drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours(100, 8000, printDebug = True),
    DetectShapes(printResult = True)
])
GaussianBlur_V_B = 0
def GaussianBlur_B(newVal):
    if newVal < 1:
        GaussianBlur_V_B = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_B = newVal - 1
    pipeline_B.blocks[1] = AverageBlur(GaussianBlur_V_B)
def Threshold_min_B(newVal):
    Threshold_min_V = newVal
    Threshold_max_V = pipeline_B.blocks[2].upperBound
    pipeline_B.blocks[2] = Threshold(Threshold_min_V,Threshold_max_V, showOutput = True, outputWindowName='Threshold B')
    print(Threshold_min_V,'-',Threshold_max_V)
def Threshold_max_B(newVal):
    Threshold_min_V = pipeline_B.blocks[2].lowerBound
    Threshold_max_V = newVal
    pipeline_B.blocks[2] = Threshold(Threshold_min_V,Threshold_max_V, showOutput = True, outputWindowName='Threshold B')
    print(Threshold_min_V,'-',Threshold_max_V)

def ThresholdContours_min_B(newVal):
    min_V = newVal
    max_V = pipeline_B.blocks[4].maxArea
    pipeline_B.blocks[4] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)
def ThresholdContours_max_B(newVal):
    min_V = pipeline_B.blocks[4].minArea
    max_V = newVal
    pipeline_B.blocks[4] = ThresholdContours(min_V,max_V, printDebug = True)
    print(min_V,'-',max_V)


cv2.createTrackbar("Gaussian_D", "Trackbars", 1, 125, GaussianBlur_D)
cv2.createTrackbar("Gaussian_R", "Trackbars", 1, 125, GaussianBlur_R)
cv2.createTrackbar("Gaussian_G", "Trackbars", 1, 125, GaussianBlur_G)
cv2.createTrackbar("Gaussian_B", "Trackbars", 1, 125, GaussianBlur_B)
cv2.createTrackbar("Thres_minD", "Trackbars", 1, 255, Threshold_min_D)
cv2.createTrackbar("Thres_maxD", "Trackbars", 1, 255, Threshold_max_D)
cv2.createTrackbar("Thres_minR", "Trackbars", 1, 255, Threshold_min_R)
cv2.createTrackbar("Thres_maxR", "Trackbars", 1, 255, Threshold_max_R)
cv2.createTrackbar("Thres_minG", "Trackbars", 1, 255, Threshold_min_G)
cv2.createTrackbar("Thres_maxG", "Trackbars", 1, 255, Threshold_max_G)
cv2.createTrackbar("Thres_minB", "Trackbars", 1, 255, Threshold_min_B)
cv2.createTrackbar("Thres_maxB", "Trackbars", 1, 255, Threshold_max_B)
cv2.createTrackbar("Conto_minD", "Trackbars", 1, 2000, ThresholdContours_min_D)
cv2.createTrackbar("Conto_maxD", "Trackbars", 1, 2000, ThresholdContours_max_D)
cv2.createTrackbar("Conto_minR", "Trackbars", 1, 2000, ThresholdContours_min_R)
cv2.createTrackbar("Conto_maxR", "Trackbars", 1, 2000, ThresholdContours_max_R)
cv2.createTrackbar("Conto_minG", "Trackbars", 1, 2000, ThresholdContours_min_G)
cv2.createTrackbar("Conto_maxG", "Trackbars", 1, 2000, ThresholdContours_max_G)
cv2.createTrackbar("Conto_minB", "Trackbars", 1, 2000, ThresholdContours_min_B)
cv2.createTrackbar("Conto_maxB", "Trackbars", 1, 2000, ThresholdContours_max_B)

while(True):
    frameD = cap.getDepthFrame()

    frame = cap.getPreviewFrame()

    cv2.imshow("frameD", frameD)
    cv2.imshow("frame", frame)

    pipeRes_R = pipeline_R.run(frame)
    
    pipeRes_G = pipeline_G.run(frame)

    pipeRes_B = pipeline_B.run(frame)

    pipeRes_D = pipeline_D.run(frameD)

    shapeImg_D = frame.copy()
    for s in pipeRes_D :
        cv2.drawContours(shapeImg_D, [s.contour], -1, (255, 255, 255), 2)
        cv2.putText(shapeImg_D, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    #cv2.imshow("Shapes D", shapeImg_D)
    shapeImg_R = shapeImg_D.copy()
    for s in pipeRes_R :
        cv2.drawContours(shapeImg_R, [s.contour], -1, (0, 0, 255), 2)
        cv2.putText(shapeImg_R, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,255), 2)
    #cv2.imshow("Shapes R", shapeImg_R)
    shapeImg_G = shapeImg_R.copy()
    for s in pipeRes_G :
        cv2.drawContours(shapeImg_G, [s.contour], -1, (0, 255, 0), 2)
        cv2.putText(shapeImg_G, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,255,150), 2)
    #cv2.imshow("Shapes G", shapeImg_G)
    shapeImg_B = shapeImg_G.copy()
    for s in pipeRes_B :
        cv2.drawContours(shapeImg_B, [s.contour], -1, (255, 0, 0), 2)
        cv2.putText(shapeImg_B, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,150,150), 2)
    cv2.imshow("Shapes", shapeImg_B)
    #shapeImg = frame.copy()
    #for s in pipeRes_D,pipeRes_B,pipeRes_G,pipeRes_R :
    #    cv2.drawContours(shapeImg, [s.contour], -1, (255, 255, 255), 2)
    #    cv2.putText(shapeImg, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    #    cv2.drawContours(shapeImg, [s.contour], -1, (255, 0, 0), 2)
    #    cv2.putText(shapeImg, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,150,150), 2)
    #    cv2.drawContours(shapeImg, [s.contour], -1, (0, 255, 0), 2)
    #    cv2.putText(shapeImg, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,255,150), 2)
    #    cv2.drawContours(shapeImg, [s.contour], -1, (0, 0, 255), 2)
    #    cv2.putText(shapeImg, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,255), 2)

    #cv2.imshow("Shapes", shapeImg)
    key = cv2.waitKey(15)

    if key == ord('q'):
        break

    elif key == ord('t'):
        cap.triggerAutoFocus()

    elif key == ord('a'):
        cap.startContinousAutoFocus()


cv2.destroyAllWindows()
cv2.destroyAllWindows()