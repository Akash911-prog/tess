from threading import Thread as T

from core.lcn import LCN
from core.state_manager import StateManager
from core.stt import STT

class Main():

    def __init__(self) -> None:
        self.normalizer = LCN()
        self.state_manager = StateManager()
        self.stt = STT(state_manager=self.state_manager)
        self.indicator = self.stt.indicator

    def run(self) -> None:
        text = ''
        while text.lower() != "close":
            print("ready")
            text = self.stt.start_listening()
            self.state_manager.set('processing')
            print(text)
            print("processing")
            result = self.normalizer.normalize(text)
            print(result)
            self.state_manager.set('idle')
            print("idle")
        self.stt.close()


if __name__ == "__main__":
    main = Main()
    main.run()