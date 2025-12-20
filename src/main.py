from threading import Thread as T
import logging

from core.lcn import LCN
from core.state_manager import StateManager
from core.stt import STT
from libs.logger_config import setup_logging


setup_logging(
    log_level="DEBUG",
    log_dir="logs",
    enable_console=True,
    enable_file=True
)

logger = logging.getLogger('tess.main')

class Main():

    def __init__(self) -> None:
        self.normalizer = LCN()
        self.state_manager = StateManager()
        self.stt = STT(state_manager=self.state_manager)
        self.indicator = self.stt.indicator

    def run(self) -> None:
        logger.info("ready")
        text = ''
        while text.lower() != "close":
            logger.info("ready")
            text = self.stt.start_listening()
            self.state_manager.set('processing')
            logger.info(text)
            logger.info("processing")
            result = self.normalizer.normalize(text)
            logger.info(result)
            self.state_manager.set('idle')
            logger.info("idle")
        self.stt.close()


if __name__ == "__main__":
    main = Main()
    main.run()
