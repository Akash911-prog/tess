class StateManager:
    def __init__(self):
        self.state = None

    def set(self, state):
        self.state = state
    
    def get(self):
        return self.state