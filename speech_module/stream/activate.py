import pvcobra
import pvporcupine
import time
import os
import struct
import numpy as np
from pydub import AudioSegment
import sys
import queue
import threading 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from AssistantGlasses.speech_module.stream.utils import voice_to_text,keyword_check
from dotenv import load_dotenv

def detection(ak,st,ss,streaming,audio,keys,q:queue.Queue | None=None):
    load_dotenv()
    cobra=pvcobra.create(access_key=ak,device="best") 
    print("Agent in command...")
    if isinstance(audio,pvporcupine.Porcupine):
        print("Starting realtime recognition...")
        rate=audio.sample_rate
        length=audio.frame_length
        text=loop3(st,ss,cobra,audio,streaming,length,rate,q,keys) # add audio input if using loop3
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
    blank=True
    while True:
        #print(streaming.is_active())
        pcm_bytes=streaming.read(length)
        pcm_ints=struct.unpack_from("h"*length,pcm_bytes)
        audio_buffer.append(np.frombuffer(pcm_bytes,dtype=np.int16))
        speech_detection=cobra.process(pcm_ints)
        if speech_detection>float(SPEECH_SENSITIVITY):
            blank=False
            last_detection_time=time.time()
        if time.time()-last_detection_time>float(SILENCE_THRESHOLD):
            break
    audio_buffer=np.concatenate(audio_buffer)
    if not blank:
        text=voice_to_text(audio_buffer,rate)
    else:
        text="Standing by..."
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
        # play and check if audio is loaded successfully, delete the next line after checking
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

""" 
    loop3: inner loop for realtime speech2text + voice termination
    BE AWARE!!! This loop may be filled with bugs as it has NOT BEEN TESTED
"""

def loop3(st,ss,cobra,audio,streaming,length,rate,q,keys):
    audio_data=streaming.read(audio.frame_length,exception_on_overflow=False)
    last_detection_time=time.time()
    load_dotenv()
    audio_buffer=[]
    blank=True
    flush=""
    key_thread=threading.Thread(target=keyword_check, args=(audio, audio_data))
    key_thread.start()
    stt_thread=None
    while True:
        try:
            key_val=keys.get_nowait()
            if key_val in ["to_agent", "navigation"]:
                break
        except:
            pass
        #print(streaming.is_active())
        pcm_bytes=streaming.read(length)
        pcm_ints=struct.unpack_from("h"*length,pcm_bytes)
        audio_buffer.append(np.frombuffer(pcm_bytes,dtype=np.int16))
        speech_detection=cobra.process(pcm_ints)
        if speech_detection>float(ss):
            blank=False
            last_detection_time=time.time()
        if time.time()-last_detection_time>float(st):
            if not blank:
                audio_buffer=np.concatenate(audio_buffer)
                if stt_thread is None or not stt_thread.is_alive():
                    stt_thread=threading.Thread(target=voice_to_text,args=(audio_buffer,rate,q))
                    stt_thread.start()
                audio_buffer=[]
                blank=True
        try:
            stt_result=q.get_nowait()
            if stt_result:
                flush+=stt_result
        except:
            pass
    return flush
"""
如果不是“结束”就只存到input的queue
"""