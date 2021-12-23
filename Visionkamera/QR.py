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
    X = int((A[0]+B[0])/2)
    Y = int((A[1]+B[1])/2)
    xta = AX/AY
    xv = math.degrees(math.atan(xta))
    yta = AY/AX
    yv = math.degrees(math.atan(yta))
    return AX,AY,Ar,xv,yv

def Distance_img(img,A = (0,0),B =(0,0)):
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

def XYCO(P1,A1,P2,A2):
    A = (P1*-1) + A1
    B = (P1*-1) + A1 + P2 + A2
    C = A / B
    O = A1 + C
    return C,O

def XYMM(m1,m2,XY):
    m,m,PX,XV,YV = Distance(m1[1],m2[1])
    V = XV,YV
    A = (m1[1][XY]*-1) + m1[2][XY]
    B = m1[2][XY] + PX + m2[2][XY]
    C = A / B
    O = m1[2][XY] + C

    print('m1',m1)
    print('m2',m2)
    print('A',A,' B',B,' V',V,' C',C,' O',O,' XY',XY)
    return O,C,V[XY]

def QR(img):
    nr = 0
    Robot = []
    Data = []
    DataA = []
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
        #print(QR_XY)
        myData2 = myData.split(',')
        if myData2[0] == 'Robot':
            Robot.append([int(myData2[1]),XY])
        else:
            Data.insert(-1,[myData,XY])
    cv2.imshow('Result QR',img)
    return Data,Robot


class Omregning:

    def __init__(self,navn) -> None:
        self.Robot_O = [0,0,0,0,0,0]
        self.Robotnr = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        file = open(navn , "r")     
        data5 = file.read().split("\n")
        for nr in data5:
            data4 = nr.split(",")
            self.Robotnr[int(data4[0])] = [int(data4[1]),int(data4[2])]
        print('Robotnr',self.Robotnr)
      
    def Nulstilling(self,Robot):
        Robotnr = self.Robotnr
        Robot_O = self.Robot_O
        nr1 = 0
        for m1 in Robot:
            nr2 = 0
            for m2 in Robot:
                if Robotnr[m1[0]] != Robotnr[m2[0]] and int(nr1) < int(nr2):
                    print(Robotnr[m1[0]],Robotnr[m2[0]],nr1,nr2)
                    if  int(Robotnr[m1[0]][0]) ==  int(Robotnr[m2[0]][0]):
                        Axx = (((m1[1][1])+ Robotnr[m1[0]][1])/((m1[1][1])+ Robotnr[m1[0]][1]+m2[1][1]-Robotnr[m2[0]][1]))
                        Axy = 0#(((m1[1][0])+ Robotnr[m1[0]][1])/((m1[1][0])+ Robotnr[m1[0]][1]+m2[1][0]-Robotnr[m2[0]][1]))
                        Cx = Robotnr[m1[0]][1] + Axx #+ Axy
                        Robot_O[0] = Axx
                        Robot_O[1] = Axy
                        Robot_O[2] = Cx
                    elif int(Robotnr[m1[0]][1]) ==  int(Robotnr[m2[0]][1]):
                        Ayx = (((m1[1][1])+ Robotnr[m1[0]][0])/((m1[1][1])+ Robotnr[m1[0]][0]+m2[1][1]-Robotnr[m2[0]][0]))
                        Ayy = 0#(((m1[1][0])+ Robotnr[m1[0]][0])/((m1[1][0])+ Robotnr[m1[0]][0]+m2[1][0]-Robotnr[m2[0]][0]))
                        Cy =Robotnr[m1[0]][0] + Ayy #+ Ayx
                        Robot_O[3] = Ayx
                        Robot_O[4] = Ayy
                        Robot_O[5] = Cy
                nr2 += 1
            nr1 += 1
        print('Axx Axy Cx Ayx Ayy Cy')
        print(Robot_O)    
        #print(nr2)
        self.Robot_O = Robot_O
        return Robot_O

    def Omregning(self,P_XY):
        Axx = self.Robot_O[0]
        Axy = self.Robot_O[1]
        Cx = self.Robot_O[2]
        Ayx = self.Robot_O[3]
        Ayy = self.Robot_O[4]
        Cy = self.Robot_O[5]
        P_X = P_XY[0]
        P_Y = P_XY[1]
        X = Cx + P_X * Axx + P_Y * Axy
        Y = Cy + P_Y * Ayy + P_X * Ayx
        print('Axx',Axx,'Axy',Axy,'Cx',Cx,'P_X',P_X,'X',X)
        print('Ayx',Ayx,'Ayy',Ayy,'Cy',Cy,'P_y',P_Y,'y',Y)
        return [X,Y]
        