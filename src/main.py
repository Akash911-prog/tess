from core.lcn import LCN
from core.state_manager import StateManager
from core.stt import STT

class Main():

    def __init__(self) -> None:
        self.lcn = LCN()
        self.state_manager = StateManager()
        self.stt = STT(state_manager=self.state_manager)
        self.indicator = self.stt.indicator

    def run(self, text: str) -> None:
        pass


if __name__ == "__main__":
    main = Main()
    main.run("open file")