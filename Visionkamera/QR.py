
import cv2
import numpy as np
from pyzbar.pyzbar import decode
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
        self.KaliCounter = 0
        self.BE_t = False
        self.Robot_O = [0,0,0,0]
        self.Robotnr = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        file = open(navn , "r")
        data6 = file.read().split("\n//")
        self.data_B = data6[1].split(",")     
        data5 = data6[0].split("\n")
        self.angle = 0
        self.rotering = [0,0,0,0]
        self.nulstilling = False
        self.Break1 = False
        self.Break2 = False
        self.Break3 = False
        self.BE_t_2 = False
        for nr in data5:
            data4 = nr.split(",")
            self.Robotnr[int(data4[0])] = [float(data4[1]),float(data4[2])]
        print('Robotnr',self.Robotnr)
      
    def Nulstilling(self,img):
        QR_img = img.copy()
        if self.Break3==False:
            print('Break3',self.Break3)
            self.Rotering_T(QR_img)
        
        elif self.Break3 == True and self.BE_t_2 == False:
            print('Break2',self.Break3,self.BE_t_2)
            QR_img = self.Rotering(img.copy())
            self.Beskere(QR_img)
        
        elif self.BE_t == True and self.Break3 == True:
            print('Break1',self.Break3,self.BE_t_2)
            QR_img = self.Rotering(img.copy())
            self.Omregning_T(QR_img,)

        
        Break = False
        if self.Break1 and self.Break2 and self.Break3 and self.BE_t:
            self.nulstilling = True
            Break = True

        if (self.Break1 and self.Break2 and self.Break3 and self.BE_t) or self.KaliCounter >= 60:
            print(self.Robot_O)
            self.KaliCounter = 0
            self.Break1 = False
            self.Break2 = False
            self.Break3 = False
            self.BE_t_2 = False
        self.KaliCounter += 1    
        
        print('')
        print ('X',self.Break2,'Y',self.Break1,'V',self.Break3,'B',self.BE_t,'Break',Break,'nr',self.KaliCounter)
        print('')
        return QR_img,Break

    def Rotering_T(self,img):
        Data,Robot,QR_img1 = QR(img)
        
        #center = (w / 2, h / 2)
        #scale = 1.0
        print(Robot)
        
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
                    print (CM_Y,CM_X)
                    P_C = RO_A_P_Y - RO_B_P_Y
                    P_B = RO_A_P_X - RO_B_P_X
                    P_A = ((P_B)**2+(P_C)**2)**0.5
                    print('P_A',P_A,'P_B',P_B,'P_C',P_C)
                    print()
                    if  RO_A_CM_X ==  RO_B_CM_X:
                        V_A = math.tan( P_B / P_C )*180/math.pi
                        V_B = math.tan( P_C / P_B  )*180/math.pi
                        print('A',V_A,'B',V_B)
                        

                        if CM_Y >=0:
                            if P_C >= P_B:
                                self.angle = -V_A 
                                Break = True
                                print('0 angle A:',V_A,'angle:',self.angle)
                                break
                            elif P_C < P_B:
                                self.angle = -V_A 
                                Break = True
                                print('1 angle A:',V_A,'angle:',self.angle)
                                break
                        elif CM_Y <0:
                            if P_C >= P_B:
                                self.angle = -V_A +180
                                Break = True
                                print('2 angle A:',V_A,'angle:',self.angle)
                                break
                            elif P_C < P_B:
                                self.angle = -V_B +180
                                Break = True
                                print('3 angle A:',V_B,'angle:',self.angle)
                                break
                    nr2 +=1
            nr1 +=1
            self.Break3 = False
        img_2 = self.Rotering(img)    
        Data,Robot,QR_img1 = QR(img_2)
        nr1 = 0
        for m2 in Robot:
            nr2 = 0
            if self.Break3:
                break
            for m1 in Robot:
                if self.Break3:
                    break
                print(Robotnr[m1[0]],Robotnr[m2[0]],nr1,nr2,m1[1][0],m2[1][0],m1[0],m2[0])
                if Robotnr[m1[0]] != Robotnr[m2[0]] and int(nr1) < int(nr2):
                    if Robotnr[m1[0]][0]==Robotnr[m2[0]][0]:
                        self.Break3 = m1[1][0] < m2[1][0] +10 and 10 + m1[1][0] > m2[1][0]
                        print(self.Break3,m1[1][0],m2[1][0])
                print(self.Break3,'F')        
                nr2 += 1
            nr1 +=1
      
    def Beskere (self,img):
        (w, h) = img.shape[:2]
        Robotnr = self.Robotnr
        Data,Robot,QR_img = QR(img)
        self.BE = [0,0,0,0]
        self.BE_t = False 
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
                        print(self.Break3, m1[1][0],m2[1][0])
                        QR_img = cv2.line(QR_img,m2[1],m1[1],(0,0,255),4)
                        self.rotering[0] = m2[1]
                        self.rotering[1] = m1[1]
                        Break1 = True
                        print(m2[1][1],m1[1][1],Robotnr[m2[0]][1],Robotnr[m1[0]][1])
                        F = m2[1][1]-m1[1][1]
                        B = Robotnr[m2[0]][1]-Robotnr[m1[0]][1]
                        C = B/F
                        A = -C * m2[1][1] + Robotnr[m2[0]][1]
                        L = (float(self.data_B[3])/C - A)
                        R = (float(self.data_B[1])/C - A)
                        print(float(self.data_B[3]),L)
                        print(float(self.data_B[1]),R)
                        print(A,C,B,F,'Y',L,R)
                        if L <0: L = 0
                        if R <L+100: R = L+100
                        if R >w: R = w
                        if L >R+50: L = R+50
                        
                        
                        self.BE[0] = int(L)
                        self.BE[1] = int(R)
                        print(A,C,B,F,'Y',L,R,'F')
                    elif Robotnr[m1[0]][1]==Robotnr[m2[0]][1]:
                        QR_img = cv2.line(QR_img,m2[1],m1[1],(0,255,0),4)
                        self.rotering[2] = m2[1]
                        self.rotering[3] = m1[1]
                        Break2 = True
                        print(m2[1][0],m1[1][0],Robotnr[m2[0]][0],Robotnr[m1[0]][0])
                        F = m2[1][0]-m1[1][0]
                        B = Robotnr[m2[0]][0]-Robotnr[m1[0]][0]
                        C = B/F
                        A = -C * m2[1][0] + Robotnr[m2[0]][0]
                        T = (float(self.data_B[0])/C - A)
                        B = (float(self.data_B[2])/C - A)
                        print(float(self.data_B[0]),T)
                        print(float(self.data_B[2]),B)
                        print(A,C,B,F,'x',T,B)
                        if T <0: T = 0
                        if B <T+100: B = T+100
                        if B >h: B = h
                        if T >B+50: T = B+50
                        self.BE[2] = int(T)
                        self.BE[3] = int(B)
                        print(A,C,B,F,'x',T,B,'F')
                nr2 +=1
            nr1 +=1
        self.BE_t = Break1 and Break2
        self.BE_t_2 = Break1 and Break2

    def Omregning_T (self,img):
        Data,Robot,QR_img = QR(img)
        Robotnr = self.Robotnr
        self.Break1 = False
        self.Break2 = False
        nr1 = 0
        for m2 in Robot:
            nr2 = 0
            if self.Break1 and self.Break2:
                        break
            for m1 in Robot:
                if self.Break1 and self.Break2:
                    break
                if Robotnr[m1[0]] != Robotnr[m2[0]] and int(nr1) < int(nr2):
                    
                    if Robotnr[m1[0]][0]==Robotnr[m2[0]][0]:
                        QR_img = cv2.line(QR_img,m2[1],m1[1],(0,0,255),4)
                        self.rotering[0] = m2[1]
                        self.rotering[1] = m1[1]
                        self.Break1 = True
                        print(m2[1][1],m1[1][1],Robotnr[m2[0]][1],Robotnr[m1[0]][1])
                        F = m2[1][1]-m1[1][1]
                        B = Robotnr[m2[0]][1]-Robotnr[m1[0]][1]
                        C = B/F
                        A = -C * m2[1][1] + Robotnr[m2[0]][1]

                        self.Robot_O[3] = A
                        self.Robot_O[2] = C
                        print(A,C,B,F,'Y')
                    elif Robotnr[m1[0]][1]==Robotnr[m2[0]][1]:
                        QR_img = cv2.line(QR_img,m2[1],m1[1],(0,255,0),4)
                        self.rotering[2] = m2[1]
                        self.rotering[3] = m1[1]
                        self.Break2 = True
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

    def Rotering(self,img):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        M = cv2.getRotationMatrix2D(center, self.angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))
        if self.BE_t:
            print(self.BE)
            rotated = rotated[self.BE[0]:self.BE[1],self.BE[2]:self.BE[3]]
              
        return rotated
    
    def Rotering_m(self,img):
        if self.nulstilling:
            img = cv2.line(img,self.rotering[0],self.rotering[1],(0,0,255),4)
            img = cv2.line(img,self.rotering[2],self.rotering[3],(0,255,0),4)
        else:
            print('Rotering XY')
        
        return img

    def Omregning(self,P_XY,img):
        cm_X = self.Robot_O[1] + P_XY[0] * self.Robot_O[0]
        cm_y = self.Robot_O[3] + P_XY[1] * self.Robot_O[2]
        print('cm_X',cm_X,'=',self.Robot_O[1],'+',P_XY[1],'*',self.Robot_O[0])
        print('cm_Y',cm_y,'=',self.Robot_O[3],'+',P_XY[0],'*',self.Robot_O[2])
        cv2.putText(img,'X:'+str(int(cm_X*100)/100),(P_XY[0],P_XY[1]+25),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2 )
        cv2.putText(img,'Y:'+str(int(cm_y*100)/100),(P_XY[0],P_XY[1]+50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2 )
        return [cm_X,cm_y],img

    def Omregning_V(self,P_XY,img,Besked = ''):
        img2 = img.copy()
        print(P_XY)
        p_cm = math.sqrt(self.Robot_O[0]**2 + self.Robot_O[2]**2)
        # Klodsens position findes i cm
        X = P_XY[0][0][0] + ((P_XY[2][0][0] - P_XY[0][0][0])/2)
        Y = P_XY[1][0][1] + ((P_XY[3][0][1] - P_XY[1][0][1])/2)
        XY = int(X),int(Y)
        CM_XY,img = self.Omregning(XY,img)
        # Finder vinkel på hvor meget klodsen drejet i X
        M1 = P_XY[0][0][0] - P_XY[1][0][0]
        M2 = P_XY[0][0][1] - P_XY[1][0][1]
        Px = math.sqrt(M1**2 + M2**2)
        VX = math.sin( M1 / Px )*180/math.pi

        Px_CM = math.sqrt((M2*self.Robot_O[2])**2 + (M1*self.Robot_O[0])**2)
        print('p_cm:',p_cm,'\n','Px:',Px,'M1:',M1,'M2:',M2,'cm:',Px_CM,'V:',VX)
        # Finder vinkel på hvor meget klodsen drejet i Y
        M3 = P_XY[0][0][0] - P_XY[2][0][0]
        M4 = P_XY[0][0][1] - P_XY[2][0][1]
        Py = math.sqrt(M3**2 + M4**2)
        VY = math.sin( M3 / Py )*180/math.pi

        Py_CM = math.sqrt((M4*self.Robot_O[2])**2 + (M3*self.Robot_O[0])**2)
        print('Px:',Py,'M1:',M3,'M2:',M4,'cm:',Py_CM,'V:',VY,'\n','XY:',XY)
        # Hvilken vej skal griberen dreje
        if Px <= Py:
            V = VX
            print('V',V,'_x')
        else:
            V = VY
            print('V',V,'_y')
        #Her lægges grafik på billedet
        cv2.drawContours(img2, [P_XY], -1, (255, 255, 255), 5)
        cv2.putText(img2,'V:'+str(int(V*10)/10)+' X:'+str(int(CM_XY[0]*10)/10)+' Y:'+str(int(CM_XY[1]*10)/10),(XY[0],XY[1]+75),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2 )
        cv2.putText(img2,'PX:'+str(int(X))+' PY:'+str(int(Y)),(XY[0],XY[1]+100),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2 )
        cv2.putText(img2,'cm X:'+str(int(Px_CM*10)/10)+' Y:'+str(int(Py_CM*10)/10),(XY[0],XY[1]+125),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2 )
        cv2.putText(img2,'p X:'+str(int(Px))+' Y:'+str(int(Py)),(XY[0],XY[1]+150),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2 )
        cv2.putText(img2,str(Besked),(XY[0],XY[1]+175),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2 )
        return CM_XY,V,img2

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