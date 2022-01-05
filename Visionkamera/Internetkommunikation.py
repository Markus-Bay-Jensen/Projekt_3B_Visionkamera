import socket
import cv2
import imutils
import numpy as np
import base64
import time




class TCP_pi_Server_M:
    
    def __init__(self, HOST = '', PORT = 50007,AntalKlienter = 1) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        self.s.listen(AntalKlienter)
        self.AntalKlienter = AntalKlienter
        self.conn = []
        #self.conn[AntalKlienter] = None
        #self.addr[AntalKlienter] = None
    
    def TCP_Aben (self, ID = 0):
        self.conn, addr = self.s.accept()
        #print('Connection from: ', addr, ' ID: ',ID)
        return self.conn,addr
        
    def TCP_Send(self, data, ID = 0):
        data2 = str(data) 
        msgToSendInBinary = data2.encode('ascii')
        self.conn.sendall(msgToSendInBinary)    

    def TCP_Modtaget (self, ID = 0, L = 4096,TCP_Luk = 1):
        data = self.conn.recv(L)
        data2 = data.decode('ascii')
        if data.decode('ascii') == 'TCP_Luk'and TCP_Luk == 1:
            self.s.close()
            data2 ='TCP_Luk'
            
        data2 = data.decode('ascii')
        print('data: ',data2)
        return data2

    def TCP_Send_Modtaget(self,data, ID = 0, L = 4096,TCP_Luk = 1):    
        data2 = str(data) 
        msgToSendInBinary = data2.encode('ascii')
        self.conn.sendall(msgToSendInBinary)
        b = cv2.waitKey(1000)
        data = self.conn.recv(L)
        data2 = data.decode('ascii')
        if data.decode('ascii') == 'TCP_Luk'and TCP_Luk == 1:
            self.s.close()
            data2 ='TCP_Luk'
            
        print('data: ',data2)
        return data2

    def TCP_Luk(self, ID = 0):
        data = 'TCP_Luk'
        msgToSendInBinary = data.encode('ascii')
        self.conn.sendall(msgToSendInBinary)
        self.conn.close()
        

    def TCP_Luk_Luk(self, ID = 0):
        
        self.s.close()

class TCP_pi_Server:
    
    def __init__(self, HOST = '', PORT = 50007) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((HOST, PORT))
        self.s.listen(1)
        self.AntalKlienter = 1
        self.conn = []
        #self.conn[AntalKlienter] = None
        #self.addr[AntalKlienter] = None
    
    def TCP_Aben (self, ID = 0):
        self.conn, addr = self.s.accept()
        print('Connection from: ', self.addr, ' ID: ',ID)
        return self.conn[ID],addr
        

    def TCP_Modtaget (self, ID = 0, L = 4096,TCP_Luk = 1):
        data = self.conn.recv(L)
        data2 = data.decode('ascii')
        if data.decode('ascii') == 'TCP_Luk'and TCP_Luk == 1:
            self.s.close()
            data2 ='TCP_Luk'
            
        data2 = data.decode('ascii')
        print('data: ',data2)
        return data2

    def TCP_Send_Modtaget(self,data, ID = 0, L = 4096,TCP_Luk = 1):    
        data2 = str(data) 
        msgToSendInBinary = data2.encode('ascii')
        self.conn.sendall(msgToSendInBinary)
        b = cv2.waitKey(1000)
        data = self.conn.recv(L)
        data2 = data.decode('ascii')
        if data.decode('ascii') == 'TCP_Luk'and TCP_Luk == 1:
            self.s.close()
            data2 ='TCP_Luk'
            
        print('data: ',data2)
        return data2

    def TCP_Send_T(self, data = [], ID = 0):
        m2 = 0
        for m in data:
            msgToSendInBinary = msgToSendInBinary + m.to_bytes(2, 'big')
            m2 += 1
        msgToSendInBinary = m2.to_bytes(2, 'big') + msgToSendInBinary
        print(msgToSendInBinary) 
        self.conn.sendall(msgToSendInBinary)
        
    def TCP_Send(self, data, ID = 0):
        data2 = str(data) 
        msgToSendInBinary = data2.encode('ascii')
        self.conn.sendall(msgToSendInBinary)

    def TCP_Luk(self, ID = 0):
        data = 'TCP_Luk'
        msgToSendInBinary = data.encode('ascii')
        self.conn.sendall(msgToSendInBinary)
        self.conn.close()
        

    def TCP_Luk_Luk(self, ID = 0):
        
        self.s.close()

class TCP_pi_Klient:
    
    def __init__(self, HOST = '127.0.0.1', PORT = 50007) -> None:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.HOST = HOST
        self.PORT = PORT

    def TCP_Aben (self):
        self.s.connect((self.HOST, self.PORT))

    def TCP_Modtaget (self,L = 4096,TCP_Luk = 1):
        data = self.s.recv(L)
        data2 = data.decode('ascii')
        if data.decode('ascii') == 'TCP_Luk'and TCP_Luk == 1:
            self.s.close()
            data2 ='TCP_Luk'
            
        print('data: ',data2)
        return data2

    def TCP_Send_Modtaget(self,data,L = 4096,TCP_Luk = 1):    
        data2 = str(data) 
        msgToSendInBinary = data2.encode('ascii')
        self.s.sendall(msgToSendInBinary)
        b = cv2.waitKey(1000)
        data = self.s.recv(L)
        data2 = data.decode('ascii')
        if data.decode('ascii') == 'TCP_Luk'and TCP_Luk == 1:
            self.s.close()
            data2 ='TCP_Luk'
            
        print('data: ',data2)
        return data2

    def TCP_Send(self, data):
        data2 = str(data) 
        msgToSendInBinary = data2.encode('ascii')
        self.s.sendall(msgToSendInBinary)

    def TCP_Luk(self):
        data = 'TCP_Luk'
        msgToSendInBinary = data.encode('ascii')
        self.s.sendall(msgToSendInBinary)
        self.s.close()

class UDP_pi_Send:
    
    def __init__(self, HOST = '127.0.0.1', PORT = 5005) -> None:
        self.UDP_IP = HOST
        self.Port = PORT
        self.fps,self.st,self.frames_to_count,self.cnt = (0,0,20,0)
        self.WIDTH = 400

    def UDP_Aben (self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('HOST:',self.UDP_IP,' PORT:',self.Port,'\n',self.sock)
        
    def UDP_Send(self, data):
        data2 = str(data) 
        msgToSendInBinary = data2.encode('ascii')
        self.sock.sendto(msgToSendInBinary, (self.UDP_IP, self.Port))

    def UDP_Send_video(self, video,Transmitting = True):
        freme = imutils.resize(video,width=self.WIDTH)
        encobed,buffer = cv2.imencode('.jpg',freme,[cv2.IMWRITE_JPEG_QUALITY,80])
        msgToSendInBinary = base64.b64encode(buffer)
        self.sock.sendto(msgToSendInBinary, (self.UDP_IP, self.Port))
        if Transmitting:
            cv2.imshow('Transmitting',video)
            key = cv2.waitKey(1) & 0xff

class UDP_pi_Send_M:
    
    def __init__(self, HOST_PORT) -> None:
        self.UDP_IP_Port = HOST_PORT
        self.fps,self.st,self.frames_to_count,self.cnt = (0,0,20,0)
        self.WIDTH = 400

    def UDP_Aben (self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('HOST PORT:',self.UDP_IP_Port,'\n',self.sock)
        
    def UDP_Send(self, data):
        data2 = str(data) 
        msgToSendInBinary = data2.encode('ascii')
        self.sock.sendto(msgToSendInBinary, (self.UDP_IP_Port))

    def UDP_Send_video(self, video,Transmitting = True):
        freme = imutils.resize(video,width=self.WIDTH)
        encobed,buffer = cv2.imencode('.jpg',freme,[cv2.IMWRITE_JPEG_QUALITY,80])
        msgToSendInBinary = base64.b64encode(buffer)
        self.sock.sendto(msgToSendInBinary, (self.UDP_IP_Port))
        if Transmitting:
            cv2.imshow('Transmitting',video)
            key = cv2.waitKey(1) & 0xff

class UDP_pi_Modtaget:
    
    def __init__(self, HOST = '127.0.0.1', PORT = 5005) -> None:
        self.HOST = HOST
        self.PORT = PORT

    def UDP_Aben (self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.HOST, self.PORT))

    def UDP_Modtaget (self,L = 4096):
        data ,addr = self.sock.recvfrom(L)
        data2 = data.decode('ascii')
                   
        print('data: ',data2,' addr: ',addr)
        return data2

    def UDP_Modtaget_video(self,L = 32768):
        msgToSendInBinary ,addr = self.sock.recvfrom(L)
        #msgToSendInBinary = str(msgToSendInBinary)
        data = base64.b64decode(msgToSendInBinary, ' /')
        npdata = np.fromstring(data,dtype=np.uint8)
        video = cv2.imdecode(npdata,1)
        
        return video

class UDP_pi_Modtaget_M:
    
    def __init__(self, HOST_PORT) -> None:
        self.HOST_PORT = HOST_PORT
        

    def UDP_Aben (self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.HOST_PORT))

    def UDP_Modtaget (self,L = 4096):
        data ,addr = self.sock.recvfrom(L)
        data2 = data.decode('ascii')
                   
        print('data: ',data2,' addr: ',addr)
        return data2

    def UDP_Modtaget_video(self,L = 32768):
        msgToSendInBinary ,addr = self.sock.recvfrom(L)
        #msgToSendInBinary = str(msgToSendInBinary)
        data = base64.b64decode(msgToSendInBinary, ' /')
        npdata = np.fromstring(data,dtype=np.uint8)
        video = cv2.imdecode(npdata,1)
        
        return video

class Video_TCP_UDP_ip_Send:
    def __init__(self , Host, Post = 50007,AntalKlienter = 10) -> None:
        self.fps,self.st,self.frames_to_count,self.cnt = (0,0,20,0)
        self.WIDTH = 400
        self.TCP =  TCP_pi_Server_M(HOST= '', PORT= Post, AntalKlienter= AntalKlienter)
        self.Host = Host
        self.m = -1

    def TCP_Aben(self ):
        self.conn,self.addr =self.TCP.TCP_Aben()
        
    def UDP_Aben(self ,kameraKode='1234',kameraKode2='ikke'):
        self.besked= self.TCP.TCP_Send_Modtaget('password:')
        if   (self.besked == kameraKode):
            self.besked= self.TCP.TCP_Send_Modtaget('kamera ID:')
            s = cv2.waitKey(2000)
            self.TCP.TCP_Send('UDP/IP')
            print(self.conn)
            print('IP : ',self.addr[0])
            print('PORT : ',self.addr[1])
            ip = int(self.addr[1])
            
            pi = self.addr[0] + str('-') + str(ip) + str('-')
            self.TCP.TCP_Send(pi)
            self.TCP.TCP_Luk()
            
            UDP = UDP_pi_Send(self.addr[0],ip)
            UDP.UDP_Aben()
            
            cap = cv2.VideoCapture(int(self.besked))

            while True:
                
            
                while True:
                
                    ret = False

                    while ret == False: 
                        ret, frame = cap.read()
                        m2 = cv2.waitKey(10)

                    cv2.imshow("Video_2", frame)
                    
                    print(m2)
                    if self.m != -1 or m2 == 113:
                        break
                    
                    
                    UDP.UDP_Send_video(frame)

                if self.m == 113 or m2 == 113:
                    break
                elif self.m >= 0 and self.m <= 9:
                    
                    cap = cv2.VideoCapture(self.m)   
                
            self.TCP.TCP_Luk()

        elif (self.besked == kameraKode2):
            self.TCP.TCP_Send('UDP/IP')
            self.besked= self.TCP.TCP_Send_Modtaget('kamera ID:')
            Port = int(self.conn[1])+1
            besked = self.conn[0]+','+Port
            self.TCP.TCP_Send(besked)
            self.UDP = UDP_pi_Send(HOST=self.conn[0],PORT=int(self.conn[1]+1))
            self.UDP.UDP_Aben()
            self.nr = 18000

        else:
            self.TCP.TCP_Send('ffff')
            self.TCP.TCP_Luk()

    def Video(self , Video,):
        self.nr -=1
        self.UDP.UDP_Send_video(Video)
        if  self.nr <=0: 
            self.nr = 3600
            self.besked = self.TCP.TCP_Modtaget(L=32)
            if self.besked != '':
                return False
            else:
                return True

                

    def TCP_Luk(self , ID = 0):
        self.TCP.TCP_Luk(ID)
        

    def TCP_Luk_Luk(self , ID = 0):
        self.TCP.TCP_Luk_Luk(ID)

class Video_TCP_UDP_ip_Modtaget:
    def __init__(self,HOST,PORT=5005) -> None:
        self.TCP = TCP_pi_Klient(HOST,int(PORT))
        self.J_HOST = HOST
        self.J_PORT = PORT
             
    def Video(self):
        self.TCP.TCP_Aben()  
        while True:
            Modtaget = self.TCP.TCP_Modtaget()
            if Modtaget =='UDP/IP':
                Modtaget=self.TCP.TCP_Modtaget()
                host,port2,m = Modtaget.split('-')
                port = int(port2)
                self.TCP.TCP_Luk()
                print (host,' - ',port)
                
                UDP = UDP_pi_Modtaget(host,port)
                UDP.UDP_Aben()
               
                while True:
                    Video = UDP.UDP_Modtaget_video()
                    cv2.imshow("Video", Video)
                    
                    s = cv2.waitKey(10)
                    print(s)
                    
                    if s == 113:
                        break
                if s == 113:
                    break

            userIn = input(Modtaget+' >>> ')

            if userIn == 'q':
                break

            self.TCP.TCP_Send(userIn)

        self.TCP.TCP_Luk()
    
         
        
