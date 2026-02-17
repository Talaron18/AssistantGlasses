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
        keyword_paths=[os.environ.get('KEYWORD_PATHS_ZH')],
        model_path=os.environ.get('MODEL_PATH_ZH')
    )
    pa=pyaudio.PyAudio()
    streaming=pa.open(
        input_device_index=1, # match the right input device
        channels=1, # porcupine requires single channel input
        rate=audio.sample_rate,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=audio.frame_length
    )
    if streaming !=None:
        print("system initiated...")
    try:
        while True:
            audio_data=streaming.read(audio.frame_length,exception_on_overflow=False)
            if wake(audio,audio_data):
                print("On your command")
                text=detection(streaming,audio)
                print(text)
    except:
        print("stop recording...")
    finally:
        # clear cache occupancy
        streaming.stop_stream()
        streaming.close()
        pa.terminate()
        audio.delete()

if __name__=="__main__":
    stream()