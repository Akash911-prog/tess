import pyaudio
import numpy as np
import pvporcupine
import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from libs.indicator import Indicator
from core.context_manager import StateManager
from configs.stt import ACCESS_KEY, KEYWORD_PATH, FORMAT, html_url

class STT():
    """
    STT (Speech-to-text) class that handles the speech recognition functionality.
    
    It uses Porcupine for wake-word detection and PyAudio for reading audio frames.
    It also uses Selenium for controlling the browser.
    """

    def __init__(self, **kwargs) -> None:
        self.state : StateManager = kwargs.get("state_manager") #type: ignore
        self.indicator = Indicator(self.state)
        self.running = True

        #==== load Procupine (wake word) =====#
        self.porcupine = pvporcupine.create(
            access_key=ACCESS_KEY, #pvporcupine access key
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


    # ---------------------------------------------------

    def read_audio(self):
        """Reads one frame and returns both PCM ndarray and raw bytes."""
        data = self.stream.read(
            self.porcupine.frame_length,
            exception_on_overflow=False
        ) # reads one frame of voice
        pcm = np.frombuffer(data, dtype=np.int16) #converts the stream of bytes to a int16 type array
        return pcm, data

    # ---------------------------------------------------

    def detect_wake_word(self, pcm):
        """Returns True if wake-word detected."""
        result = self.porcupine.process(pcm)
        return result >= 0

    # ---------------------------------------------------
    
    def stt_once(self):
        """
        STT (speech-to-text) function that triggers the browser's speech recognition functionality once.

        It clicks the start button, waits for the speech recognition to finish, and then returns the recognized text.

        :return: Recognized text as a string.
        """
        # Click the start button
        start_btn = self.driver.find_element(By.ID, 'start')
        start_btn.click()

        self.state.set("listening")

        # Wait for the speech recognition to finish
        WebDriverWait(self.driver, 30).until(
            lambda d: d.execute_script("return document.body.getAttribute('data-speech');") == "stopped"
        )

        # Get the recognized text
        element = self.driver.find_element(By.CLASS_NAME, 'text')
        result = element.text
        return result

    # ---------------------------------------------------

    def start_listening(self) -> str:
        """
        Starts the speech recognition functionality and listens for a single speech.

        It detects the wake word and then triggers the browser's speech recognition functionality once.

        :return: Recognized text as a string.
        """
        # Loop until the wake word is detected
        while self.running:
            # Read one frame of audio
            pcm, _ = self.read_audio()
            # Check if the wake word is detected
            if self.detect_wake_word(pcm):
                print("Listening...")
                # Break out of the loop
                break
        
        # Trigger the browser's speech recognition functionality once
        text = self.stt_once()
        # Return the recognized text
        return text
    
    def close(self):
        """
        Closes the speech recognition functionality and releases all associated resources.

        This method should be called when the speech recognition functionality is no longer needed.
        """

        self.running = False

        self.driver.close()
        self.driver.quit()
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        self.porcupine.delete()


if __name__ == "__main__":


    # Verify the new current working directory
    new_current_dir = os.getcwd()
    print(f"New current working directory: {new_current_dir}")
    text = ''
    stt = STT(state_manager=StateManager())

    while text.lower() != "close":
        text = stt.start_listening()
        print(text)
    
    stt.close()