import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR


cam = OAKCamColor(900,900)

pipeline = PostProcessingPipeline([
    #GetRedChannel(),
    ConvertToGray(showOutput = True, outputWindowName = 'test window'),
    #AverageBlur(7, True),
    GaussianBlur(201),
    #BilateralFilter(5, 10, 10, True)
    #LaplacianSharpen(0.5, True, True)
    #UnsharpMasking(7, True),
    Threshold(200, 250, showOutput = True),
    #AdaptiveThreshold(200, 11, 2, showOutput = True),
    #OtsuBinarization(True)
    DetectContours(draw = True, drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours(100, 8000, printDebug = True),
    DetectShapes(printResult = True)
])
O = QR.Omregning('QR.txt')

while(True):
    frame = cam.getPreviewFrame()
    Data,R= QR.QR(frame)
    O.Nulstilling(R)
    if cv2.waitKey(1) == ord('m'):
        break
while(True):
    frame = cam.getPreviewFrame()
    #Data,R= QR.QR(frame)
    #O.Nulstilling(R)
    pipeRes = pipeline.run(frame)

    shapeImg = frame.copy()
    for s in pipeRes:
        XY = O.Omregning(s.center)
        cv2.drawContours(shapeImg, [s.contour], -1, (0, 255, 0), 2)
        cv2.putText(shapeImg, str(XY), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)


    #cv2.imshow("Webcam", pipeRes)
    cv2.imshow("Shapes", shapeImg)

    

    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
