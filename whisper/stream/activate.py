import pvcobra
import time
import os
import struct
import numpy as np
from utils import wav_to_text

def detection(streaming,audio):
    cobra=pvcobra.create(access_key=os.environ.get('PORCUPINE_KEY'),device="GPU")
    audio_buffer=[]
    last_detection_time=time.time()
    rate=audio.sample_rate
    SILENCE_THRESHOLD=5.0
    SPEECH_SENSITIVITY=0.5
    print("agent in command...")
    while True:
        pcm_bytes=streaming.read(audio.frame_length)
        pcm_ints=struct.unpack_from("h"*audio.frame_length,pcm_bytes)
        audio_buffer.append(np.frombuffer(pcm_bytes,dtype=np.int16))

        speech_detection=cobra.process(pcm_ints)
        if speech_detection>SPEECH_SENSITIVITY:
            last_detection_time=time.time()
        if time.time()-last_detection_time>SILENCE_THRESHOLD:
            print("Auf Auweisungen warten! ...")
            break
    cobra.delete()
    text=wav_to_text(audio_buffer,rate)
    return text