import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import pyaudio
from AssistantGlasses.speech_module.stream.utils import wake
from AssistantGlasses.speech_module.stream.activate import detection
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
        print("System initiated...")
    try:
        while True:
            audio_data=streaming.read(audio.frame_length,exception_on_overflow=False)
            if wake(audio,audio_data):
                print("On your command")
                text=detection(streaming,audio)
                print(text)
                print("Standing by...")
    except:
        print("Stop recording...")
    finally:
        # clear cache occupancy
        streaming.stop_stream()
        streaming.close()
        pa.terminate()
        audio.delete()
"""
    add another wake-up word to end conversation
"""