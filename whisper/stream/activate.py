import pvcobra
import pvporcupine
import time
import os
import struct
import numpy as np
from pydub import AudioSegment
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from AssistantGlasses.whisper.stream.utils import voice_to_text
from dotenv import load_dotenv

def detection(streaming,audio):
    load_dotenv()
    cobra=pvcobra.create(access_key=os.environ.get('PORCUPINE_KEY'),device="best") 
    print("Agent in command...")
    if isinstance(audio,pvporcupine.Porcupine):
        print("Atarting realtime recognition...")
        rate=audio.sample_rate
        length=audio.frame_length
        text=loop1(cobra,streaming,length,rate)
    elif isinstance(audio,AudioSegment):
        print("Starting audio recognition...")
        rate=audio.frame_rate
        length=cobra.frame_length
        text=loop2(cobra,streaming,length,audio,rate)
    else:
        print(f"Unkown type:{type(audio)}")
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
        print(streaming.is_active())
        pcm_bytes=streaming.read(length)
        pcm_ints=struct.unpack_from("h"*length,pcm_bytes)
        audio_buffer.append(np.frombuffer(pcm_bytes,dtype=np.int16))
        speech_detection=cobra.process(pcm_ints)
        if speech_detection>float(SPEECH_SENSITIVITY):
            last_detection_time=time.time()
        if time.time()-last_detection_time>float(SILENCE_THRESHOLD):
            print("Standing by...")
            break
    audio_buffer=np.concatenate(audio_buffer)
    text=voice_to_text(audio_buffer,rate)
    return text

# inner loop for recorded speech2text
def loop2(cobra,streaming,length,audio,rate):
    last_detection_time=time.time()
    load_dotenv()
    SILENCE_THRESHOLD=os.environ.get('SILENCE_THRESHOLD')
    SPEECH_SENSITIVITY=os.environ.get('SPEECH_SENSITIVITY')
    data=audio.raw_data
    bytes=length*2
    for i in range(0,len(data),bytes):
        chunk=data[i:i+bytes]
        # checking if audio is loaded successfully
        streaming.write(chunk)
        if len(chunk)==bytes:
            pcm=struct.unpack_from("h"*length,chunk)
            speech_detection=cobra.process(pcm)
            if speech_detection>float(SPEECH_SENSITIVITY):
                last_detection_time=time.time()
        if time.time()-last_detection_time>float(SILENCE_THRESHOLD):
            print("Standing by...")
            break 
    audio_int16=np.frombuffer(data,dtype=np.int16)
    text=voice_to_text(audio_int16,rate)
    return text