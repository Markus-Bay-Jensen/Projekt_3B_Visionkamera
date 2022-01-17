from statemachine import State
import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
import Internetkommunikation
from pipeline import *
import math

class start:
    TCP = Internetkommunikation.TCP_pi_Server(HOST='',PORT=23)
    Robot_o = QR.Omregning('QR.txt')
    cap = OAKCamColor(1920,1080)
    F1 = 0
    F2 = 0
    F3 = 0

F = start

class Opstart(State):
    def Enter(self):
        print()
        print('Opstart')

    def Execute(self):
        print()
        frame = F.cap.getPreviewFrame()
        cv2.imshow("Firkan", frame)
        self.stateMachine.ChangeState(TCP_Connection())

class TCP_Connection(State):
    def Enter(self):
        print()
        print('TCP_Connection')


    def Execute(self):
        F.TCP.TCP_Aben()
        self.stateMachine.ChangeState(Server_listening())

class Server_listening(State):
    def Enter(self):
        print()
        print('Server_listening')
        self.Besked = 'm'
        print(self.Besked)


    def Execute(self):
        mm = cv2.waitKey(100)
        self.Besked = F.TCP.TCP_Modtaget()
        print(self.Besked,'waitKey',mm) 
        if self.Besked == 'TCP_Luk' or self.Besked == ''or mm == ord('t'):
            self.stateMachine.ChangeState(TCP_Connection())
        if self.Besked == 'Klar':
            self.stateMachine.ChangeState(Shape_Detection())
        if self.Besked == 'q' or mm == ord('q'):
            self.stateMachine.ChangeState(Close_Program())
        if self.Besked == 'Kalibrering'or mm == ord('k'):
            self.stateMachine.ChangeState(Kalibrering_QR())

class Shape_Detection(State):
    def Enter(self):
        print()
        print('Shape_Detection')

    def Execute(self):
        frame = F.cap.getPreviewFrame()
        frame = F.Robot_o.Rotering(frame)
        Firkan_2 = []
        Firkan_Liste = PipeRes(frame.copy())
        frame_Firkan =frame.copy()
        for f in Firkan_Liste:
            #print('Firkan_Liste',f)
            print(' P:',f[1].points,' F:',f[0])
            if f[1].points == 4:
                Firkan_2.append([f[1].approx,f[0]])
                print('P:',f[1].points,' F:',f[0])
            #cv2.drawContours(frame_Firkan, [f.contour], -1, (0, 255, 0), 2)
        frame_Firkan = F.Robot_o.Rotering_m(frame_Firkan)    
        cv2.imshow("Firkan", frame_Firkan)
        self.stateMachine.ChangeState(Databehandlin_Konttrol(Firkan_2,img=frame_Firkan))

class Databehandlin_Konttrol(State):
    def __init__(self,frame,img) -> None:
        self.frame = frame
        self.img = img

    def Enter(self):
        print()
        print('Databehandlin_Konttrol')
        print('F1',F.F1,'F2',F.F2,'F3',F.F3)
        #print('frame',self.frame)

    def Execute(self):
        M2 = [-0.374,0.08105,0.008,3.11,0,0,4]
        M = '['+str(M2[0])+','+str(M2[1])+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+',4]'
        M3 = M
        # Næste del er valg af hvilken farve som skal samles op
        F1_ = False
        F2_ = False
        F3_ = False
        for F_cm in self.frame:
            if F_cm[1] == 1:
                F1_ = True
            if F_cm[1] == 2:
                F2_ = True
            if F_cm[1] == 3:
                F3_ = True
        F1_3 = F1_ and ((F2_==False or F.F1 <= F.F2) and (F3_==False or F.F1 <= F.F3))
        F2_3 = F2_ and ((F1_==False or F.F2 < F.F1) and (F3_==False or F.F2 <= F.F3))
        F3_3 = F3_ and ((F1_==False or F.F3 < F.F1) and (F2_==False or F.F3 < F.F2))
        for F_cm in self.frame:
            if F1_3 and F_cm[1] == 1:
                F.F1 += 1
                XY , V , self.img = F.Robot_o.Omregning_V(F_cm[0],self.img,'1')
                print('F:',F_cm[1],'X:',XY[0],'Y:',XY[1],'V:',V)
                RV = V/180*math.pi
                cv2.imshow("Firkan", self.img)
                M = '['+str(XY[0]/100)+','+str(XY[1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(RV)+','+str(M2[5])+','+ str(F_cm[1]) +']'
                break
            if F2_3 and F_cm[1] == 2:
                F.F2 += 1
                XY , V , self.img = F.Robot_o.Omregning_V(F_cm[0],self.img,'2')
                print('F:',F_cm[1],'X:',XY[0],'Y:',XY[1],'V:',V)
                RV = V/180*math.pi
                cv2.imshow("Firkan", self.img)
                M = '['+str(XY[0]/100)+','+str(XY[1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(RV)+','+str(M2[5])+','+ str(F_cm[1]) +']'
                break
            if F3_3 and F_cm[1] == 3:
                F.F3 += 1
                XY , V , self.img = F.Robot_o.Omregning_V(F_cm[0],self.img,'3')
                print('F:',F_cm[1],'X:',XY[0],'Y:',XY[1],'V:',V)
                RV = V/180*math.pi
                cv2.imshow("Firkan", self.img)
                M = '['+str(XY[0]/100)+','+str(XY[1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(RV)+','+str(M2[5])+','+ str(F_cm[1]) +']'
                break
            if F_cm[1] >= 4 and F1_ == False and F2_ == False and F3_ == False:
                XY , V , self.img = F.Robot_o.Omregning_V(F_cm[0],self.img,'4')
                print('F:',F_cm[1],'X:',XY[0],'Y:',XY[1],'V:',V)
                RV = V/180*math.pi
                cv2.imshow("Firkan", self.img)
                M = '['+str(XY[0]/100)+','+str(XY[1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(RV)+','+str(M2[5])+','+ str(F_cm[1]) +']'
                break
        if M != M3:
            self.stateMachine.ChangeState(Serd_data_to_client(M))
        else:
            cv2.waitKey(250)
            self.stateMachine.ChangeState(Shape_Detection())

class Serd_data_to_client(State):
    def __init__(self,frame) -> None:
        self.frame = frame

    def Enter(self):
        print()
        print('Serd_data_to_client')
        print(self.frame)

    def Execute(self):
        F.TCP.TCP_Send(self.frame)
        self.stateMachine.ChangeState(Server_listening())

class Close_Program(State):
    def Enter(self):
        print()
        print('Close_Program')

    def Execute(self):
        F.TCP.TCP_Luk()
        F.TCP.TCP_Luk_Luk()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        print("Finished saving!")
        self.stateMachine.running = False

class Kalibrering_QR(State):
    def Enter(self):
        print()
        print('Kalibrering_QR')

    def Execute(self):
        frame_QR = F.cap.getPreviewFrame()
        frame_QR = pipeline_QR.run(frame_QR.copy())
        frame_QR,Break= F.Robot_o.Nulstilling(frame_QR)
        cv2.imshow("QR", frame_QR)
        print("hej")
        key = cv2.waitKey(500)
        if Break:
            if key == ord('k'):
                self.stateMachine.ChangeState(Kalibrering_F())
            else:
                if key == ord('l'):
                    pass
                else:
                    F.TCP.TCP_Send('[100]')
                    self.stateMachine.ChangeState(Server_listening())
                    
class Kalibrering_F(State):
    def Enter(self):
        print()
        print('Kalibrering_F')

    def Execute(self):
        frame = F.cap.getPreviewFrame()
        frame = F.Robot_o.Rotering(frame)
        Firkan_Liste = PipeRes(frame.copy())
        frame_Firkan =frame.copy()
        for f in Firkan_Liste:
            #print('Firkan_Liste',f)
            print(' P:',f[1].points,' F:',f[0])
            cv2.drawContours(frame_Firkan, [f[1].approx], -1, (255, 255, 255), 5)
            cv2.putText(frame_Firkan,'F:'+str(f[0]),(f[1].center[0],f[1].center[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2 )
            cv2.putText(frame_Firkan,'P:'+str(f[1].points),(f[1].center[0],f[1].center[1]+25),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2 )
        frame_Firkan = F.Robot_o.Rotering_m(frame_Firkan)    
        cv2.imshow("Firkan", frame_Firkan)
        key = cv2.waitKey(100)
        if key == ord('f'):
            self.stateMachine.ChangeState(Kalibrering_F())

