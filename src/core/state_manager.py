class StateManager:
    def __init__(self):
        self.state = 'idle'

    def set(self, state):
        self.state = state
    
    def get(self):
        return self.state.lower()
    
    def __str__(self) -> str :
        return self.state.lower()
    
if __name__ == '__main__':
    state_manager = StateManager()
    state_manager.set('listening')
    a = state_manager
    print(a == "listening")