import pyaudio
import numpy as np
import pvporcupine
import sys
from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import threading as T

from libs.indicator import Indicator
from core.state_manager import StateManager
from configs.stt import ACCESS_KEY, KEYWORD_PATH, FORMAT, html_url

class STT():
    def __init__(self, **kwargs) -> None:
        self.indicator = Indicator()
        self.state : StateManager = kwargs.get("state_manager") #type: ignore

        #==== load Procupine (wake word) =====#
        self.porcupine = pvporcupine.create(
            access_key=ACCESS_KEY,
            keyword_paths=[KEYWORD_PATH],
        )

        #==== pyaudio stream startup =====#
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            format=FORMAT,
            channels=1,
            rate=self.porcupine.sample_rate,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
        )

        #==== Selenium Driver Startup =====#
        self.options = Options()
        self.options.add_argument('--headless=new')
        self.options.add_argument("--use-fake-ui-for-media-stream")  
        self.options.add_argument("--use-fake-device-for-media-stream")
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get(html_url)


    def set_state(self, state: str) -> None:
        self.state.set(state)
