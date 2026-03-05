import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import pyaudio
from AssistantGlasses.speech_module.stream.utils import wake
from AssistantGlasses.speech_module.stream.activate import detection
from dotenv import load_dotenv
import queue
import pvporcupine

def stream(listen: queue.Queue | None=None):
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
    if listen is None:
        listen=queue.Queue()
    if streaming is not None:
        print("System initiated...")
    try:
        while True:
            audio_data=streaming.read(audio.frame_length,exception_on_overflow=False)
            if wake(audio,audio_data):
                print("On your command")
                text=detection(streaming,audio)
                listen.put(text.strip()) # adding input text to a queue, ready for update
    except Exception as e:
        print(f"Receiving interuption: {e}")
        print("Stop recording...")
    finally:
        # clear cache occupancy
        streaming.stop_stream()
        streaming.close()
        pa.terminate()
        audio.delete()
"""
    add another wake-up word to end conversation
    设置2个关键词：发送（只发送字符串，不关闭语音识别）、结束（关闭语音识别）
"""