import pvcobra
import time
import os
import struct
import numpy as np
from utils import voice_to_text
from dotenv import load_dotenv

def detection(streaming,audio):
    load_dotenv()

    # check out for available devices
    devices=pvcobra.available_devices()
    print(devices)
    cobra=pvcobra.create(access_key=os.environ.get('PORCUPINE_KEY'),device="GPU") 
    if audio.sample_rate:
        rate=audio.sample_rate
    else:
        rate=audio.frame_rate
    print("agent in command...")
    if audio.frame_length:
        length=audio.frame_length
        text=loop1(cobra,streaming,length,rate)
    else:
        length=cobra.frame_length
        text=loop2(cobra,length,audio,rate)
    cobra.delete()
    return text

# inner loop for realtime speech2text 
def loop1(cobra,streaming,length,rate):
    last_detection_time=time.time()
    load_dotenv()
    SILENCE_THRESHOLD=os.environ.get('SILENCE_THRESHOLD')
    SPEECH_SENSITIVITY=os.environ.get('SPEECH_SENSITIVITY')
    audio_buffer=[]
    while True:
        pcm_bytes=streaming.read(length)
        pcm_ints=struct.unpack_from("h"*length,pcm_bytes)
        audio_buffer.append(np.frombuffer(pcm_bytes,dtype=np.int16))
        speech_detection=cobra.process(pcm_ints)
        if speech_detection>SPEECH_SENSITIVITY:
            last_detection_time=time.time()
        if time.time()-last_detection_time>SILENCE_THRESHOLD:
            print("standing by...")
            break
    text=voice_to_text(audio_buffer,rate)
    return text

# inner loop for recorded speech2text
def loop2(cobra,length,audio,rate):
    last_detection_time=time.time()
    load_dotenv()
    SILENCE_THRESHOLD=os.environ.get('SILENCE_THRESHOLD')
    SPEECH_SENSITIVITY=os.environ.get('SPEECH_SENSITIVITY')
    data=audio.raw_data
    bytes=length*2
    for i in range(0,len(data),bytes):
        chunk=data[i:i+bytes]
        if len(chunk)==bytes:
            pcm=struct.unpack_from("h"*length,chunk)
            speech_detection=cobra.process(pcm)
            if speech_detection>SPEECH_SENSITIVITY:
                last_detection_time=time.time()
        if time.time()-last_detection_time>SILENCE_THRESHOLD:
            print("standing by...")
            break 
    audio_int16=np.frombuffer(data,dtype=np.int16)
    text=voice_to_text(audio_int16,rate)
    return text