from statemachine import State
import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
import Internetkommunikation
from pipeline import *
TCP = Internetkommunikation.TCP_pi_Server(HOST='',PORT=23)
Robot_o = QR.Omregning('QR.txt')
cap = OAKCamColorDepth(1920,1080)
F1 = 0
F2 = 0
F3 = 0

class Opstart(State):
    def Execute(self):
        self.stateMachine.ChangeState(TCP_Connection())

class TCP_Connection(State):
    def Execute(self):
        TCP.TCP_Aben()
        self.stateMachine.ChangeState(Server_listening())

class Server_listening(State):
    def Execute(self):
        Besked = TCP.TCP_Modtaget()
        if False:
            self.stateMachine.ChangeState(TCP_Connection())
        if Besked == 'Klar':
            self.stateMachine.ChangeState(Shape_Detection())
        if Besked == 'q':
            self.stateMachine.ChangeState(Close_Program())
        if Besked == 'Kalibrering':
            self.stateMachine.ChangeState(Kalibrering_QR())

class Shape_Detection(State):
    def Execute(self):
        frame = cap.getPreviewFrame()
        Firkan_cm = []
        Firkan_Liste = PipeRes(frame.copy())
        frame_Firkan =frame.copy()
        for f in Firkan_Liste:
            XY , V , frame_Firkan = Robot_o.Omregning_V(f[2],frame_Firkan)
            print('CM:',XY,' P:',f[1],'V:',V,' F:',f[0])
            Firkan_cm.append([XY,V,f[0]]) 
        cv2.imshow("Firkan", frame_Firkan)
        self.stateMachine.ChangeState(Databehandlin_Konttrol(Firkan_cm))

class Databehandlin_Konttrol(State):
    def __init__(self,frame) -> None:
        self.frame = frame

    def Execute(self):
        M2 = [-0.23339,0.02856,0.2481,0,3.0765,0,4]
        M = '['+str(M2[0])+','+str(M2[1])+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+',4]'
        M3 = M
        F1_ = False
        F2_ = False
        F3_ = False
        for F_cm in self.frame:
            if F_cm[2] == 1:
                F1_ = True
            if F_cm[2] == 2:
                F2_ = True
            if F_cm[2] == 3:
                F3_ = True
        F1_3 = F1_ and ((F2_==False or F1 <= F2) and (F3_==False or F1 <= F3))
        F2_3 = F2_ and ((F1_==False or F2 < F1) and (F3_==False or F2 <= F3))
        F3_3 = F3_ and ((F1_==False or F3 < F1) and (F2_==False or F3 < F2))
        for F_cm in self.frame:
            if F1_3 and F_cm[2] == 1:
                M = '['+str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+ str(F_cm[2]) +']'
                break
            if F2_3 and F_cm[2] == 2:
                M = '['+str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+ str(F_cm[2]) +']'
                break
            if F3_3 and F_cm[2] == 3:
                M = '['+str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+ str(F_cm[2]) +']'
                break
            if F_cm[2] >= 4 and F1_ == False and F2_ == False and F3_ == False:
                M = '['+str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+ str(F_cm[2]) +']'
                break
        if M != M3:
            self.stateMachine.ChangeState(Serd_data_to_client())
        self.stateMachine.ChangeState(Shape_Detection(M))

class Serd_data_to_client(State):
    def __init__(self,frame) -> None:
        self.frame = frame

    def Execute(self):
        TCP.TCP_Send(self.frame)
        self.stateMachine.ChangeState(Server_listening())


class Close_Program(State):
    def Execute(self):
        TCP.TCP_Luk()
        TCP.TCP_Luk_Luk()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
        print("Finished saving!")
        self.stateMachine.running = False

class Kalibrering_QR(State):
    def Execute(self):
        frame_QR = cap.getPreviewFrame()
        frame_QR = pipeline_QR.run(frame_QR.copy())
        frame_QR,Break= Robot_o.Nulstilling(frame_QR)
        cv2.imshow("QR", frame_QR)
        key = cv2.waitKey(100)
        if Break:
            TCP.TCP_Send('(100)')
            if False:
                self.stateMachine.ChangeState(Shape_Detection())
            self.stateMachine.ChangeState(Server_listening())
        
        



