# Super class for a state object
class State:
    stateMachine = None

    def Execute(self):
        pass

    def Enter(self):
        pass

    def Exit(self):
        pass


# Implementation of a state machine for running states
class StateMachine:
    def __init__(self, state):
        self.state = state
        self.state.stateMachine = self

    def Run(self):
        self.state.Enter()
        self.running = True
        while self.running:
            self.state.Execute()

    def ChangeState(self, newState):
        self.state.Exit()
        self.state = newState
        self.state.stateMachine = self
        self.state.Enter()