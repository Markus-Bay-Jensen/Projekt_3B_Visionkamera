import math
import cv2
import QR 
from OAKWrapper import *

N = QR.Omregning('QR.txt')
b = cv2.waitKey(1)
cap = OAKCamColorDepth(900,900)
while(True):
    b = cv2.waitKey(1000)
    frame = cap.getPreviewFrame()
    frame2 = N.Rotering(frame)
    
    o,o2 = N.Nulstilling(frame)
    print(o2)
    cv2.imshow("O", o)
    Qr , m ,frame2= QR.QR2(frame2)
    print(Qr)
    for m5 in Qr:
        xx,V10,frame2 = N.Omregning_V(m5[2],frame2)
    print(m)
    for m5 in m:
        xx,frame2 = N.Omregning(m5[1],frame2)
        
    cv2.imshow("webcam", frame2)
    if b == ord('q')  or b == ord('n'):
        break
cap.release()
cv2.destroyAllWindows()


