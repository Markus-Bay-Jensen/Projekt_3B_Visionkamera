from statemachine import State
import cv2
import socket


class Init(State):
    def Execute(self):
        self.stateMachine.ChangeState(Listen())

class Listen(State):
    def Execute(self):
        self.stateMachine.ChangeState(RecvCommd())

class RecvCommd(State):
    def Execute(self):
        if False:
            self.stateMachine.ChangeState(send_coord())
        if False:
            self.stateMachine.ChangeState(Listen())
        if False:
            self.stateMachine.ChangeState(Shutdown())

class send_coord(State):
    def Execute(self):
        self.stateMachine.ChangeState(Sende())

class Sende(State):
    def Execute(self):
        
        self.stateMachine.ChangeState(RecvCommd())


class Shutdown(State):
    def Execute(self):
       
        print("Finished saving!")
        self.stateMachine.running = False


        
        



