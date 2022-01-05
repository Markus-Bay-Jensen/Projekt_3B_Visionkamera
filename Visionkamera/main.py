from States import Opstart
from CustomStateMachine import *
import cv2

startingState = Opstart()
sm = MyStateMachine(startingState)
sm.Run()

print("Program closed!")