import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *

cap = OAKCamColorDepth()

pipeline_D = PostProcessingPipeline([
    GaussianBlur(1)    
    
])
pipeline_R = PostProcessingPipeline([
    GetRedChannel(),
    GaussianBlur(1)
    
])
pipeline_G = PostProcessingPipeline([
    GetGreenChannel(),
    GaussianBlur(1)    
])
pipeline_B = PostProcessingPipeline([
    GetBlueChannel(),
    GaussianBlur(1)
    
])
while(True):
    frameD = cap.getDepthFrame()

    frame = cap.getFrame()

    cv2.imshow("frameD", frameD)
    cv2.imshow("frame", frame)

    pipeRes_R = pipeline_R.run(frame)
    cv2.imshow("pipeRes_R", pipeRes_R)
    
    pipeRes_G = pipeline_G.run(frame)
    cv2.imshow("pipeRes_G", pipeRes_G)

    pipeRes_B = pipeline_B.run(frame)
    cv2.imshow("pipeRes_B", pipeRes_B)

    #pipeRes_D = pipeline_D.run(frameD)
    #cv2.imshow("pipeRes_D", pipeRes_D)

    if cv2.waitKey(2000) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
