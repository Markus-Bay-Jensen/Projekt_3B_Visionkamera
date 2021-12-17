import cv2
import numpy as np

from IMPP import *

cap = cv2.VideoCapture(0)

def nothing(x):
    pass

cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)  #--- window to have all the controls

cv2.createTrackbar("GaussianBlur", "Controls", 1, 49, nothing)
cv2.createTrackbar("LowerBound", "Controls", 1, 250, nothing)
cv2.createTrackbar("UpperBound", "Controls", 5, 255, nothing)
cv2.createTrackbar("MinArea", "Controls", 1, 30000, nothing)
cv2.createTrackbar("MaxArea", "Controls", 1, 30000, nothing)
cv2.createTrackbar("R", "Controls", 1, 255, nothing)
cv2.createTrackbar("G", "Controls", 1, 255, nothing)
cv2.createTrackbar("B", "Controls", 1, 255, nothing)



while(True):
    gaussianBlur = 2*int(cv2.getTrackbarPos("GaussianBlur", "Controls"))-1
    LowerBound =  int(cv2.getTrackbarPos("LowerBound", "Controls"))
    UpperBound =  int(cv2.getTrackbarPos("UpperBound", "Controls"))
    MinArea =  int(cv2.getTrackbarPos("MinArea", "Controls"))
    MaxArea =  int(cv2.getTrackbarPos("MaxArea", "Controls"))   
    R =  int(cv2.getTrackbarPos("R", "Controls"))
    G =  int(cv2.getTrackbarPos("G", "Controls"))
    B =  int(cv2.getTrackbarPos("B", "Controls"))
    R_Tændt =  cv2.getTrackbarPos("Tændt Slukket", "R")
    
    pipeline = PostProcessingPipeline([
        ConvertToGray(showOutput = R_Tændt, outputWindowName = 'test window'),
        GaussianBlur(gaussianBlur,R_Tændt),
        Threshold(lowerBound=LowerBound, upperBound=UpperBound, showOutput = R_Tændt),
        DetectContours(draw = R_Tændt, drawInfo = ContourDrawInfo((R, G, B), 2)),
        ThresholdContours(minArea=MinArea, maxArea=MaxArea, printDebug = R_Tændt),
        DetectShapes(printResult = R_Tændt)
    ])


    ret, frame = cap.read()

    pipeRes = pipeline.run(frame)

    shapeImg = frame.copy()
    for s in pipeRes:
        cv2.drawContours(shapeImg, [s.contour], -1, (0, 255, 0), 2)
        cv2.putText(shapeImg, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Shapes", shapeImg)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.destroyAllWindows()