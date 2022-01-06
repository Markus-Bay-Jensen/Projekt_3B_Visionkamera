from States import Opstart
from statemachine import *
import cv2

startingState = Opstart()
sm = StateMachine(startingState)
sm.Run()

print("Program closed!")