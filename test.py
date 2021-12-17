from QR import QR
import cv2


cap = cv2.VideoCapture(0)

ret, frame = cap.read()

n= QR(frame)
cv2.waitKey(0)

print(n)
