import pyaudio
from os import getenv
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()


FRAME = 512
RATE = 16000
FORMAT = pyaudio.paInt16
CHANNEL = 1
ACCESS_KEY = getenv("PVPINE_ACCESS_KEY")

current_path = Path(__file__).parent.parent
index_file_path = current_path.rglob("index.html").__next__().resolve()
KEYWORD_PATH = current_path.rglob("hey-tess_en_windows_v3_0_0.ppn").__next__().resolve()
html_url = index_file_path.as_uri()

if __name__ == "__main__":
    print(KEYWORD_PATH)