import QR
import cv2
from OAKWrapper import *
O = QR.Omregning('QR.txt')
cam = OAKCamColor(900,900)

frame = cam.getPreviewFrame()
Data,R= QR.QR(frame)
Robot = O.Nulstilling(R)

while True:
    frame = cam.getPreviewFrame()

    #Data,R= QR.QR(frame)
    #Robot = O.Nulstilling(R)

    key = cv2.waitKey(500)

    if key == ord('q'):
        break

    elif key == ord('t'):
        cam.triggerAutoFocus()

    elif key == ord('a'):
        cam.startContinousAutoFocus()

print(Robot)
