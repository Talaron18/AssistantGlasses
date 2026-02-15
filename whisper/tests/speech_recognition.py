from ..stream.activate import detection
from pydub import AudioSegment as pd
import os
import pyaudio
from dotenv import load_dotenv

def speech_test(path):
    audio=pd.from_file(path,format="m4a")
    audio=audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    pa=pyaudio.PyAudio()
    stream=pa.open(
        format=pa.get_format_from_width(audio.sample_width),
        channels=audio.channels,
        rate=audio.frame_rate,
        output=True
    )
    if stream!=None:
        print("system ready...")
    try:
        text=detection(stream,audio)
        print(text)
    except:
        print("exiting...")
    finally:
        stream.stop_stream()
        stream.close()


if __name__=="__main__":
    load_dotenv()
    test_path=os.environ.get("TEST_DIR")
    speech_test(test_path)