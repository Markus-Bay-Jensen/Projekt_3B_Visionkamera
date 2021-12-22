import sys, os
import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image
import math

def Distance(A = (0,0),B =(0,0)):
    AX = A[0]-B[0]
    AY = A[1]-B[1]
    print('fY:',AY,'fX',AX)
    if AX < 0:
        AX = AX*-1
    if AY < 0:
        AY = AY*-1
    Ar = int(math.sqrt((AX**2)+(AY**2)))
    print('Y:',AY,'X',AX,'A',Ar)
    return AX,AY,Ar,[A[0],A[1]],[B[0],B[1]]


cap = cv2.VideoCapture(0)
R1 = 0,0
R2 = 0,0
R3 = 0,0
R4 = 0,0

while True:

    ret, frame = cap.read()
    #decoded = frame
    '''
    decoded = decode(frame, symbols=[ZBarSymbol.QRCODE])
    qr_dic = {}
    for qr in decoded:
        x = qr[2][0] # The Left position of the QR code
        qr_dic[x] = qr[0] # The Data stored in the QR code

    for qr in sorted(qr_dic.keys()):
        #print(qr_dic[qr])
        pass
    '''

    img=frame
    for barcode in decode(img):
        pts2 = barcode.rect
        myData = barcode.data.decode('utf-8')
        pts = np.array(barcode.polygon,np.int32)
        pts = pts.reshape((-1,1,2))
        X = pts[0] + ((pts[2] - pts[0])/2)
        Y = pts[1] + ((pts[3] - pts[1])/2)
        XY = ((X + Y)/2)
        XY = XY[0]
        XY = int(XY[0]),int(XY[1])
        print('Data',myData,'x',XY[0],'y',XY[1])
        cv2.polylines(img,[pts],True,(0,0,255),2)
        
        cv2.putText(img,myData,(XY),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255),2 )
        QR_XY = myData,XY
        print(QR_XY)
        if QR_XY[0] == 'Red':
            R1 = QR_XY[1]
        elif QR_XY[0] == 'Bottom left':
            R2 = QR_XY[1]
        elif QR_XY[0] == 'Bottom left':
            R3 = QR_XY[1]
        elif QR_XY[0] == 'Bottom left':
            R4 = QR_XY[1]
    B,B,R_1_2,p5,p6 = Distance(R1,R2)
    X = int((p5[0]+p6[0])/2)
    Y = int((p5[1]+p6[1])/2)

    cv2.putText(img,str(R_1_2),(X,Y),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),2 )
    cv2.line(img,p5,p6,(0,0,0),2)
    #B,B,R_1_3 = Distance(R1,R3)
    #B,B,R_1_4 = Distance(R1,R4)
    #B,B,R_2_3 = Distance(R2,R3)
    #B,B,R_2_4 = Distance(R2,R4)
    #B,B,R_3_4 = Distance(R3,R4)
    print(R_1_2,X,Y)
    cv2.imshow('Result',img)
    if cv2.waitKey(0) ==ord('q'):
        break
        
