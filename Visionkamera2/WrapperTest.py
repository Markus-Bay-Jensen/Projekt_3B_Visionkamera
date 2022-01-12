from OAKWrapper import *
import cv2

#cam = OAKCamColor()
cam = OAKCamColorDepth()

while(True):
    frame = cam.getFrame()
    cv2.imshow("Color Frame", frame)

    framep = cam.getPreviewFrame()
    cv2.imshow("Preview Frame", framep)

    frame = cam.getDepthFrame()
    cv2.imshow("Depth Frame", frame)

    framec = cam.getDepthFrameColorMapped()
    cv2.imshow("Depth Frame (color mapped to JET)", framec)

    key = cv2.waitKey(15)

    if key == ord('q'):
        break

    elif key == ord('t'):
        cam.triggerAutoFocus()

    elif key == ord('a'):
        cam.startContinousAutoFocus()



