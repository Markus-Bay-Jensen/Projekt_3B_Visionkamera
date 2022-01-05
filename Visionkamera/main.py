from States import LoadData
from CustomStateMachine import *
import cv2

startingState = LoadData()
sm = MyStateMachine(startingState)
sm.Run()

print("Program closed!")