import pyaudio
from utils import wake
from activate import detection
import os
from dotenv import load_dotenv
import pvporcupine

def stream():
    load_dotenv()
    audio=pvporcupine.create(
        access_key=os.environ.get('PORCUPINE_KEY'),
        keyword_paths=[os.environ.get('KEYWORD_PATHS')],
        model_path=os.environ.get('MODEL_PATH')
    )
    pa=pyaudio.PyAudio()
    streaming=pa.open(
        channels=1,
        rate=audio.sample_rate,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=audio.frame_length
    )
    print("system initiated...")
    try:
        while True:
            audio_data=streaming.read(audio.frame_length,exception_on_overflow=False)
            if wake(audio,audio_data):
                print("On your command")
                #text=detection(streaming,audio)
                #print(text)
    finally:
        streaming.stop_stream()
        streaming.close()
        pa.terminate()
        audio.delete()

if __name__=="__main__":
    stream()