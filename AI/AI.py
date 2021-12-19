#from platform import node
import math
import cv2


class Data:
    def __init__(self,file) -> None:
        self.navn = str(file) + '.txt'

    def LoadData(self):
        data = []
        print('Abner fil "',self.navn,' "')
        file = open( self.navn , "r")     
        data5 = file.read().split("\n")
        for nr in data5:
            data4 = nr.split(".")
            data3 = [int(data4[0]) , int(data4[1]) , int(data4[2])]
            data.append(data3)
        return data

    def SaveData(self,data):
        print('Gem fil "',self.navn,' "')
        file = open(self.navn, "w")
        Gem = ""
        nr3 = 0
        for nr in data:
            if nr3 != 0:
                Gem += "\n"
            for nr2 in nr:
                Gem += str(nr2) + "."
            nr3 +=1
        file.write(Gem)
        print(Gem)

class Distance:
    def __init__(self,Farve,D = 10) -> None:
        self.Farve = Farve
        self.data = Data(Farve)
        self.D = D
        

        
    def Afstand(self,Input = [0,0,0]):
        Farve2 = self.data.LoadData()
        
        data = []
        for Farve3 in Farve2:
            R = Input[0]-Farve3[0]
            G = Input[1]-Farve3[1]
            B = Input[2]-Farve3[2]
            if R < 0:
                R *= -1
            if G < 0:
                G *= -1
            if B < 0:
                B *= -1 
            A = math.sqrt((R**2)+(G**2)+(B**2)) 
            

            
            data1 = [A , self.Farve]
            data.append(data1)
        
        data.sort()
        
        return data[0:self.D]

class AI:
    def __init__(self,input,D=10) -> None:
        self.input = input
        self.D = int(D)
        self.data = []
        self.farve = []

    def Farve (self,Farve):
        distance = Distance(Farve,self.D)
        data2 = distance.Afstand(self.input)
        self.data.extend(data2)
        self.farve.extend(Farve)

    def ai (self):
        self.data.sort()
        nr3 = 0
        Resultat2 = []
        for nr in self.farve:
            
            M =0
            for nr2 in range(self.D):
                data2 = self.data[nr2]
                if nr == data2[1]:
                    M +=1
                
            Resultat2.append([nr,M])
            nr3 +=1
        print('Resultat',Resultat2)
        g = 0
        Resultat = 'Ingen Resultat'
        for nr4 in Resultat2:
            print(nr4)
            if nr4[1] > g:
                g = nr4[1]
                Resultat = nr4[0]
            elif nr4[1] == g:
                Resultat ='Double Resultat'
        return Resultat

class Farve_tilfojelse:
    def __init__(self,Farve) -> None:
        self.data = Data(Farve)
        self.file = self.data.LoadData()
        
    def Farve(self,Farve):
        
        self.file.append(Farve)
        self.data.SaveData(self.file)
                

    
