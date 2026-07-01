import pyaudio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from AssistantGlasses.speech_module.stream.utils import wake
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
    for i in range(pa.get_device_count()):
        print(pa.get_device_info_by_index(i))
    input("Press Enter to continue...")
    print("\n")
    print(pa.get_default_input_device_info())
    input("Press Enter to continue")
    streaming=pa.open(
        input_device_index=1,
        channels=1,
        rate=audio.sample_rate,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=audio.frame_length
    )
    if streaming !=None:
        print("System initiated...")
    try:
        while True:
            audio_data=streaming.read(audio.frame_length,exception_on_overflow=False)
            if wake(audio,audio_data):
                print("On your command")
    except:
        print("Stop recording...")
    finally:
        streaming.stop_stream()
        streaming.close()
        pa.terminate()
        audio.delete()

if __name__=="__main__":
    stream()