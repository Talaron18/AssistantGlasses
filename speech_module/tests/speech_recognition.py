import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import pvcobra
from AssistantGlasses.speech_module.stream.activate import detection
from pydub import AudioSegment as pd
import os
import pyaudio
from dotenv import load_dotenv

def speech_test(path):
    # multi-type file
    if path.endswith(".m4a"):
        audio=pd.from_file(path,format="m4a")
    elif path.endswith(".mp3"):
        audio=pd.from_file(path,format="mp3")
    elif path.endswith(".wav"):
        audio=pd.from_file(path,format="wav")
    else:
        print("Unknown error...")
    # adapt to porcupine supported format
    audio=audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    pa=pyaudio.PyAudio()
    stream=pa.open(
        format=pa.get_format_from_width(audio.sample_width),
        channels=audio.channels,
        rate=audio.frame_rate,
        output=True
    )
    # check out for streaming initiation
    if stream!=None and audio!=None:
        print("System ready...")
    else:
        print("Streaming failed...")
    try:
        # check out for available devices
        devices=pvcobra.available_devices()
        print(devices)
        text=detection(stream,audio)
        print(f'Text generated: {text}')
    except Exception as e:
        print("Operation suspended...")
        print(f"Error occurred:{e}")
        import traceback
        traceback.print_exc()
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


if __name__=="__main__":
    load_dotenv()
    test_path=os.environ.get("TEST_EN")
    # checking input file path
    print("Vocal file found..." if bool(test_path) else "Path not found...")
    speech_test(test_path)
