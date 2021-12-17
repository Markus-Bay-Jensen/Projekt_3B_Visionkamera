import sys, os
import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from PIL import Image
import math

def Distance(img,A = (0,0),B =(0,0)):
    AX = A[0]-B[0]
    AY = A[1]-B[1]
    print('fY:',AY,'fX',AX)
    if AX < 0:
        AX = AX*-1
    if AY < 0:
        AY = AY*-1
    Ar = int(math.sqrt((AX**2)+(AY**2)))
    print('Y:',AY,'X',AX,'A',Ar)
    X = int((A[0]+B[0])/2)
    Y = int((A[1]+B[1])/2)

    cv2.putText(img,str(Ar),(X,Y),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0),3 )
    cv2.line(img,A,B,(255,255,255),2)
    
    return AX,AY,Ar,img

def QR(img):
    nr = 0
    Robot = []
    Data = []
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
        myData2 = myData.split(',')
        if myData2[0] == 'Robot':
            Robot.append(myData2[1],XY,[int(myData2[2]),int(myData2[3]),int(myData2[4])])
        else:
            Data.insert(-1,[myData,XY])
    nr2 = 0
    for nr in Robot:
        if nr2 != 0:
            pass

        print(Robot[nr])
        nr2 += 1
    print(nr2)
    cv2.imshow('Result',img)
    return Data
        