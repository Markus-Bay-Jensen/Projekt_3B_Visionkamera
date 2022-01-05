
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import math

'''Finpudsrotering af billedet
Finpuds omregningen
Find ud af hvor mange grader klodsen venner'''

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

def QR(frame):
    img = frame.copy()
    Robot = []
    Data = []
    for barcode in decode(img):
        pts2 = barcode.rect
        #print(pts2)
        myData = barcode.data.decode('utf-8')
        pts = np.array(barcode.polygon,np.int32)
        pts = pts.reshape((-1,1,2))
        X = pts[0][0][0] + ((pts[2][0][0] - pts[0][0][0])/2)
        Y = pts[1][0][1] + ((pts[3][0][1] - pts[1][0][1])/2)
        #print(X,Y,pts)
        XY = int(X),int(Y)
        #print('Data',myData,'x',XY[0],'y',XY[1])
        cv2.polylines(img,[pts],True,(0,0,255),2)
        
        cv2.putText(img,myData,(XY),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255),2 )
        QR_XY = myData,XY
        #print(QR_XY)
        myData2 = myData.split(',')
        if myData2[0] == 'Robot':
            Robot.append([int(myData2[1]),XY])
        else:
            Data.insert(-1,[myData,XY])
    #cv2.imshow('Result QR',img)
    return Data,Robot,img     

def QR2(frame):
    img = frame.copy()
    Robot = []
    Data = []
    for barcode in decode(img):
        pts2 = barcode.rect
        print(pts2)
        myData = barcode.data.decode('utf-8')
        pts1 = np.array(barcode.polygon,np.int32)
        pts = pts1.reshape((-1,1,2))
        X = pts[0] + ((pts[2] - pts[0])/2)
        Y = pts[1] + ((pts[3] - pts[1])/2)
        XY = int(X),int(Y)
        print('Data',myData,'x',XY[0],'y',XY[1])
        cv2.polylines(img,[pts],True,(0,0,255),2)
        
        cv2.putText(img,myData,(XY),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255),2 )
        QR_XY = myData,XY
        #print(QR_XY)
        myData2 = myData.split(',')
        if myData2[0] == 'Robot':
            Robot.append([int(myData2[1]),XY])
        else:
            Data.insert(-1,[myData,XY,pts1])
    #cv2.imshow('Result QR',img)
    return Data,Robot,img     

class Omregning:

    def __init__(self,navn) -> None:
        self.Robot_O = [0,0,0,0]
        self.Robotnr = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        file = open(navn , "r")     
        data5 = file.read().split("\n")
        self.angle = 0
        
        for nr in data5:
            data4 = nr.split(",")
            self.Robotnr[int(data4[0])] = [int(data4[1])/100,int(data4[2])/100]
        print('Robotnr',self.Robotnr)
      
    def Nulstilling(self,img):
        QR_img = img.copy()
        Data,Robot,QR_img1 = QR(QR_img)
        (w, h) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        
        
        Robotnr = self.Robotnr
        Break = False

        nr1 = 0
        for m1 in Robot:
            nr2 = 0
            if Break:
                        break
            for m2 in Robot:
                if Robotnr[m1[0]] != Robotnr[m2[0]] :#and int(nr1) < int(nr2):
                    if Break:
                        break
                    RO_A_P_X = m1[1][0]
                    RO_A_P_Y = m1[1][1]
                    RO_B_P_X = m2[1][0]
                    RO_B_P_Y = m2[1][1]

                    RO_A_CM_X = Robotnr[m1[0]][0]
                    RO_B_CM_X = Robotnr[m2[0]][0]
                    RO_A_CM_Y = Robotnr[m1[0]][1]
                    RO_B_CM_Y = Robotnr[m2[0]][1]
                    CM_Y = RO_A_CM_Y - RO_B_CM_Y
                    CM_X = RO_A_CM_X - RO_B_CM_X
                    #print (CM_Y,CM_X)
                    P_C = RO_A_P_Y - RO_B_P_Y
                    P_B = RO_A_P_X - RO_B_P_X
                    P_A = ((P_B)**2+(P_C)**2)**0.5
                    #print('P_A',P_A,'P_B',P_B,'P_C',P_C)
                    V_A = math.tan( P_C / P_B )*180/math.pi
                    if  RO_A_CM_X ==  RO_B_CM_X:
                        if CM_Y >=0:
                            if P_B >=0:
                                self.angle = (V_A)-90
                                Break = True
                                #print('0 angle A:',V_A,'angle:',self.angle)
                                break
                            elif P_B <0:
                                self.angle = V_A*-1+180-90
                                Break = True
                                #print('1 angle A:',V_A,'angle:',self.angle)
                                break

                        elif CM_Y <0:
                            if P_B >=0:
                                self.angle = V_A-90
                                Break = True
                                #print('2 angle A:',V_A,'angle:',self.angle)
                                break
                            elif P_B <0:
                                self.angle = V_A*-1-90
                                Break = True
                                #print('3 angle A:',V_A,'angle:',self.angle)
                                break
                    nr2 +=1
            nr1 +=1
        QR_img = self.Rotering(img.copy())
        Data,Robot,QR_img = QR(QR_img)
        X_t = False
        Y_t = False
        Break1 = False
        Break2 = False
        nr1 = 0
        for m2 in Robot:
            nr2 = 0
            if Break1 and Break2:
                        break
            for m1 in Robot:
                if Break1 and Break2:
                    break
                if Robotnr[m1[0]] != Robotnr[m2[0]] and int(nr1) < int(nr2):
                    
                    if Robotnr[m1[0]][0]==Robotnr[m2[0]][0]:
                        #print(m1,'-',m2)
                        #print(Robotnr[m1[0]],'-',Robotnr[m2[0]])
                        #print('M1 - M2')
                        QR_img = cv2.line(QR_img,m2[1],m1[1],(0,0,255),4)
                        Break1 = True
                        print(m2[1][1],m1[1][1],Robotnr[m2[0]][1],Robotnr[m1[0]][1])
                        F = m2[1][1]-m1[1][1]
                        B = Robotnr[m2[0]][1]-Robotnr[m1[0]][1]
                        C = B/F
                        A = -C * m2[1][1] + Robotnr[m2[0]][1]

                        self.Robot_O[3] = A
                        self.Robot_O[2] = C
                        print(A,C,B,F,'Y')


                    elif Robotnr[m1[0]][1]==Robotnr[m2[0]][1]:
                        #print(m1,'-',m2)
                        #print(Robotnr[m1[0]],'-',Robotnr[m2[0]])
                        #print('M1 - M2')
                        QR_img = cv2.line(QR_img,m2[1],m1[1],(0,255,0),4)
                        Break2 = True
                        print(m2[1][0],m1[1][0],Robotnr[m2[0]][0],Robotnr[m1[0]][0])
                        F = m2[1][0]-m1[1][0]
                        B = Robotnr[m2[0]][0]-Robotnr[m1[0]][0]
                        C = B/F
                        A = -C * m2[1][0] + Robotnr[m2[0]][0]

                        self.Robot_O[1] = A
                        self.Robot_O[0] = C
                        print(A,C,B,F,'x')
                nr2 +=1
            nr1 +=1
        print ('X',Break2,'Y',Break1)
        Break = False
        if Break1 and Break2:
            print(self.Robot_O)
            Break = True
        print('')
        return QR_img,Break

    def Omregning(self,P_XY,img):
        cm_y = self.Robot_O[1] + P_XY[0] * self.Robot_O[0]
        cm_X = self.Robot_O[3] + P_XY[1] * self.Robot_O[2]
        print('cm_X',cm_X,'=',self.Robot_O[1],'+',P_XY[1],'*',self.Robot_O[0])
        print('cm_Y',cm_y,'=',self.Robot_O[3],'+',P_XY[0],'*',self.Robot_O[2])
        cv2.putText(img,'X:'+str(int(cm_X*100)/100),(P_XY[0],P_XY[1]+25),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2 )
        cv2.putText(img,'Y:'+str(int(cm_y*100)/100),(P_XY[0],P_XY[1]+50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2 )
        return [cm_X,cm_y],img

    def Rotering(self,img):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, self.angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        return rotated
    
    def Rotering_m(self,img):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, -self.angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))
        
        return rotated

    def Omregning_V(self,P_XY,img):
        #print(P_XY)
        X = P_XY[0][0][0] + ((P_XY[2][0][0] - P_XY[0][0][0])/2)
        Y = P_XY[1][0][1] + ((P_XY[3][0][1] - P_XY[1][0][1])/2)
        XY = int(X),int(Y)
        CM_XY,img = self.Omregning(XY,img)
        M1 = P_XY[0][0][0] - P_XY[1][0][0]
        M2 = P_XY[0][0][1] - P_XY[1][0][1]
        Px = math.sqrt(M1**2 + M2**2)
        M3 = P_XY[0][0][0] - P_XY[2][0][0]
        M4 = P_XY[0][0][1] - P_XY[2][0][1]
        Py = math.sqrt(M3**2 + M4**2)
        if Px >= Py:
            V = math.sin( M3 / Py )*180/math.pi
            print(M3,M4,Py,V,'0')
        else:
            V = math.sin( M1 / Px )*180/math.pi
            print(M1,M2,Px,V,'1')
        cv2.putText(img,'V:'+str(int(V*10)/10),(XY[0],XY[1]+75),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2 )
        return CM_XY,V,img

class file:

    def __init__(self,navn) -> None:
        self.data = {}
        file = open(navn , "r")     
        data5 = file.read().split("\n")
        
        for nr in data5:
            data4 = nr.split(":")
            if data4 != '':
                self.data[data4[0]] = data4[1]
        print('data',self.data)
        return self.data

    def Gem (self,data):
        self.data =data
        for m in self.data:
            print(m)