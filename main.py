import cv2
import numpy as np

from IMPP import *

cap = cv2.VideoCapture(0)

pipeline = PostProcessingPipeline([
    #GetRedChannel(),
    ConvertToGray(showOutput = True, outputWindowName = 'test window'),
    #AverageBlur(7, True),
    GaussianBlur(201),
    #BilateralFilter(5, 10, 10, True)
    #LaplacianSharpen(0.5, True, True)
    #UnsharpMasking(7, True)
    Threshold(200, 250, showOutput = True),
    #AdaptiveThreshold(200, 11, 2, showOutput = True),
    #OtsuBinarization(True)
    DetectContours(draw = True, drawInfo = ContourDrawInfo((0, 0, 255), 2)),
    ThresholdContours(100, 8000, printDebug = True),
    DetectShapes(printResult = True)
])

while(True):
    ret, frame = cap.read()

    pipeRes = pipeline.run(frame)

    shapeImg = frame.copy()
    for s in pipeRes:
        cv2.drawContours(shapeImg, [s.contour], -1, (0, 255, 0), 2)
        cv2.putText(shapeImg, str(s.points), s.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

    cv2.imshow("Webcam", frame)
    cv2.imshow("Shapes", shapeImg)

    if cv2.waitKey(0) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
