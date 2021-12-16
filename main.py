import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
cv2.namedWindow("Trackbars")
cap = OAKCamColorDepth(640,400)

pipeline_D = PostProcessingPipeline([
    GaussianBlur(1),
    Threshold(1, 255, showOutput = True)    
    
])
GaussianBlur_V_D = 0
def GaussianBlur_D(newVal):
    if newVal < 1:
        GaussianBlur_V_D = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_D = newVal - 1
    pipeline_D.blocks[0] = AverageBlur(GaussianBlur_V_D, True)
def Threshold_min_D(newVal):
    Threshold_min_V = newVal
    Threshold_max_V = pipeline_D.blocks[1].upperBound
    pipeline_D.blocks[1] = Threshold(Threshold_min_V,Threshold_max_V)
    print(Threshold_min_V,'-',Threshold_max_V)
def Threshold_max_D(newVal):
    Threshold_min_V = pipeline_D.blocks[1].lowerBound
    Threshold_max_V = newVal
    pipeline_D.blocks[1] = Threshold(Threshold_min_V,Threshold_max_V)
    print(Threshold_min_V,'-',Threshold_max_V)

pipeline_R = PostProcessingPipeline([
    GetRedChannel(),
    GaussianBlur(1),
    Threshold(1, 255, showOutput = True)   
])
GaussianBlur_V_R = 0
def GaussianBlur_R(newVal):
    if newVal < 1:
        GaussianBlur_V_R = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_R = newVal - 1
    pipeline_R.blocks[1] = AverageBlur(GaussianBlur_V_R, True)
def Threshold_min_R(newVal):
    Threshold_min_V_R = newVal
    Threshold_max_V_R = pipeline_R.blocks[2].upperBound
    pipeline_R.blocks[2] = Threshold(Threshold_min_V_R,Threshold_max_V_R)
    print(Threshold_min_V_R,'-',Threshold_max_V_R)
def Threshold_max_R(newVal):
    Threshold_min_V_R = pipeline_R.blocks[2].lowerBound
    Threshold_max_V_R = newVal
    pipeline_R.blocks[2] = Threshold(Threshold_min_V_R,Threshold_max_V_R)
    print(Threshold_min_V_R,'-',Threshold_max_V_R)

pipeline_G = PostProcessingPipeline([
    GetGreenChannel(),
    GaussianBlur(1),
    Threshold(1, 255, showOutput = True)    
])
GaussianBlur_V_G = 0
def GaussianBlur_G(newVal):
    if newVal < 1:
        GaussianBlur_V_G = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_G = newVal - 1
    pipeline_G.blocks[1] = AverageBlur(GaussianBlur_V_G, True)
def Threshold_min_G(newVal):
    Threshold_min_V = newVal
    Threshold_max_V = pipeline_G.blocks[2].upperBound
    pipeline_G.blocks[2] = Threshold(Threshold_min_V,Threshold_max_V)
    print(Threshold_min_V,'-',Threshold_max_V)
def Threshold_max_G(newVal):
    Threshold_min_V = pipeline_G.blocks[2].lowerBound
    Threshold_max_V = newVal
    pipeline_G.blocks[2] = Threshold(Threshold_min_V,Threshold_max_V)
    print(Threshold_min_V,'-',Threshold_max_V)

pipeline_B = PostProcessingPipeline([
    GetBlueChannel(),
    GaussianBlur(1),
    Threshold(1, 255, showOutput = True)
])
GaussianBlur_V_B = 0
def GaussianBlur_B(newVal):
    if newVal < 1:
        GaussianBlur_V_B = newVal
    if newVal % 2 == 0:
        GaussianBlur_V_B = newVal - 1
    pipeline_B.blocks[1] = AverageBlur(GaussianBlur_V_B, True)
def Threshold_min_B(newVal):
    Threshold_min_V = newVal
    Threshold_max_V = pipeline_B.blocks[2].upperBound
    pipeline_B.blocks[2] = Threshold(Threshold_min_V,Threshold_max_V)
    print(Threshold_min_V,'-',Threshold_max_V)
def Threshold_max_B(newVal):
    Threshold_min_V = pipeline_B.blocks[2].lowerBound
    Threshold_max_V = newVal
    pipeline_B.blocks[2] = Threshold(Threshold_min_V,Threshold_max_V)
    print(Threshold_min_V,'-',Threshold_max_V)

cv2.createTrackbar("GaussianBlur_D", "Trackbars", 1, 125, GaussianBlur_D)
cv2.createTrackbar("GaussianBlur_R", "Trackbars", 1, 125, GaussianBlur_R)
cv2.createTrackbar("GaussianBlur_G", "Trackbars", 1, 125, GaussianBlur_G)
cv2.createTrackbar("GaussianBlur_B", "Trackbars", 1, 125, GaussianBlur_B)
cv2.createTrackbar("Threshold_min_D", "Trackbars", 1, 255, Threshold_min_D)
cv2.createTrackbar("Threshold_max_D", "Trackbars", 1, 255, Threshold_max_D)
cv2.createTrackbar("Threshold_min_R", "Trackbars", 1, 255, Threshold_min_R)
cv2.createTrackbar("Threshold_max_R", "Trackbars", 1, 255, Threshold_max_R)
cv2.createTrackbar("Threshold_min_G", "Trackbars", 1, 255, Threshold_min_G)
cv2.createTrackbar("Threshold_max_G", "Trackbars", 1, 255, Threshold_max_G)
cv2.createTrackbar("Threshold_min_B", "Trackbars", 1, 255, Threshold_min_B)
cv2.createTrackbar("Threshold_max_B", "Trackbars", 1, 255, Threshold_max_B)
while(True):
    frameD = cap.getDepthFrame()

    frame = cap.getPreviewFrame()

    cv2.imshow("frameD", frameD)
    cv2.imshow("frame", frame)

    pipeRes_R = pipeline_R.run(frame)
    cv2.imshow("pipeRes_R", pipeRes_R)
    
    pipeRes_G = pipeline_G.run(frame)
    cv2.imshow("pipeRes_G", pipeRes_G)

    pipeRes_B = pipeline_B.run(frame)
    cv2.imshow("pipeRes_B", pipeRes_B)

    pipeRes_D = pipeline_D.run(frameD)
    cv2.imshow("pipeRes_D", pipeRes_D)

    key = cv2.waitKey(15)

    if key == ord('q'):
        break

    elif key == ord('t'):
        cap.triggerAutoFocus()

    elif key == ord('a'):
        cap.startContinousAutoFocus()


cv2.destroyAllWindows()
cv2.destroyAllWindows()