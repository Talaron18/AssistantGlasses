import pyaudio
import os
from dotenv import load_dotenv
from pydub import AudioSegment as pd

def quicktest(mode="OFF"):
    if mode=="ON":
        load_dotenv()
        path=os.environ.get("TEST_MUSIC")
        if path:
            print(path)
        else:
            print("Path not found...")
        audio=pd.from_file(path,format="m4a")
        audio=audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        pa=pyaudio.PyAudio()
        stream=pa.open(
            format=pa.get_format_from_width(audio.sample_width),
            channels=audio.channels,
            rate=audio.frame_rate,
            output=True
        )
        try:
            length=1024
            data=audio.raw_data
            bytes=length*2
            for i in range(0,len(data),bytes):
                chunk=data[i:i+bytes]
                # play and check if audio is loaded successfully, delete the next line after checking
                stream.write(chunk)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
    else:
        print("Failed to activate tool...")
    return "you are fooled"

if __name__=="__main__":
    out=quicktest(mode="ON")
    print(out)