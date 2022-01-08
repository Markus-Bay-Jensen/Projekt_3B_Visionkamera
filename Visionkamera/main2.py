import cv2
import numpy as np
from OAKWrapper import *
from IMPP import *
import QR
import Internetkommunikation
from pipeline import *


TCP = Internetkommunikation.TCP_pi_Server(HOST='',PORT=23)
F1 = 0
F2 = 0
F3 = 0
Robot_o = QR.Omregning('QR.txt')
cap = OAKCamColor(1080,1080)
qr = True
qr2 = False
Firkan = False
tcp_IP = False
m100 =0
if tcp_IP:
    TCP.TCP_Aben()

while(True):
    frame = cap.getPreviewFrame()
    frame = Robot_o.Rotering(frame)
    #print('')

    if tcp_IP:
            Besked = TCP.TCP_Modtaget()
            if Besked == 'q':
                break
            elif Besked == 't':
                cap.triggerAutoFocus()
            elif Besked == 'a':
                cap.startContinousAutoFocus()

    if qr:
        
        
            

        while(True):
            
            frame_QR = cap.getPreviewFrame()
            frame_QR = pipeline_QR.run(frame_QR.copy())
            frame_QR,Break= Robot_o.Nulstilling(frame_QR)
            cv2.imshow("QR", frame_QR)
            key = cv2.waitKey(500)
            if Break:
                break
        #break
        qr = False
        Firkan = True

    if qr2:
        Data_QR,Robot = QR.QR(frame)
        for mm in Data_QR:
            XY10 = Robot_o.Omregning(mm[1])
            cv2.putText(frame,str(XY10),(mm[1][0],mm[1][1]+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2 )
            cv2.putText(frame,str(mm[1]),(mm[1][0],mm[1][1]+40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2 )
        for mm in Robot:
            XY10 = Robot_o.Omregning(mm[1])
            cv2.putText(frame,str(XY10),(mm[1][0],mm[1][1]+20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2 )
            cv2.putText(frame,str(mm[1]),(mm[1][0],mm[1][1]+40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2 )
            

    if Firkan:
        m100 +=1
        Firkan =True
        Firkan_cm = []
        Firkan_Liste = PipeRes(frame.copy())
        frame_Firkan =frame.copy()
        for f in Firkan_Liste:
            print(' P:',f[1].points,' F:',f[0])
            if f[1].points == 4:
                XY , V , frame_Firkan = Robot_o.Omregning_V(f[1].approx,frame_Firkan,'m')
                print('CM:',XY,' P:',f[1].points,'V:',V,' F:',f[0])
                Firkan_cm.append([XY,V,f[0]]) 
                break
        #cv2.imshow("Firkan", frame_Firkan)
        print(m100)
        frame = frame_Firkan

    if tcp_IP:
        M2 = [-0.23339,0.02856,0.2481,0,3.0765,0,4]
        M = '['+str(M2[0])+','+str(M2[1])+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+',4]'
        F1_ = False
        F2_ = False
        F3_ = False
        '''
        for F_cm in Firkan_cm:
            if F_cm[2] == 1:
                F1_ = True
            if F_cm[2] == 2:
                F2_ = True
            if F_cm[2] == 3:
                F3_ = True
        F1_3 = F1_ and ((F2_==False or F1 <= F2) and (F3_==False or F1 <= F3))
        F2_3 = F2_ and ((F1_==False or F2 < F1) and (F3_==False or F2 <= F3))
        F3_3 = F3_ and ((F1_==False or F3 < F1) and (F2_==False or F3 < F2))
        for F_cm in Firkan_cm:
            if F1_3 and F_cm[2] == 1:
                M  = str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+str(F1_3[2])
                break
            if F2_3 and F_cm[2] == 2:
                M  = str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+str(F1_3[2])
                break
            if F3_3 and F_cm[2] == 3:
                M  = str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+str(F1_3[2])
                break
            if F_cm[2] >= 4 and F1_ == False and F2_ == False and F3_ == False:
                M  = str(F1_3[0][0]/100)+','+str(F1_3[0][1]/100)+','+str(M2[2])+','+str(M2[3])+','+str(M2[4])+','+str(M2[5])+','+str(F1_3[2])
                break'''
        TCP.TCP_Send(M)

    cv2.imshow("frame", frame)

    k = cv2.waitKey(2000)
    if k == ord('q'):
        break
    elif k == ord('t'):
        cap.triggerAutoFocus()
    elif k == ord('a'):
        cap.startContinousAutoFocus()

#TCP.TCP_Luk()
#TCP.TCP_Luk_Luk()
cv2.destroyAllWindows()
cv2.destroyAllWindows()