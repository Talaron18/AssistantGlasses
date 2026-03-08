import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import pyaudio
from AssistantGlasses.speech_module.stream.utils import voice_to_text
from dotenv import load_dotenv
import queue
import threading
import numpy as np
import struct
import time
import pvcobra
import pvporcupine

def stream(listen_q,action_q):
    load_dotenv()
    keyword_paths=[
        os.environ.get('KEYWORD_PATHS_INIT'),
        os.environ.get('KEYWORD_PATHS_SHOT'),
        os.environ.get('KEYWORD_PATHS_NAVI_END'),
        os.environ.get('KEYWORD_PATHS_SEND'),
        os.environ.get('KEYWORD_PATHS_NAVI_START')
    ]
    for p in keyword_paths :
        if p is None:
            print("Path not found")
            return 1
    keyword_paths = [p for p in keyword_paths if p is not None]
    audio=pvporcupine.create(
        access_key=os.environ.get('PORCUPINE_KEY'),
        keyword_paths=keyword_paths,
        model_path=os.environ.get('MODEL_PATH_ZH')
    )
    SILENCE_THRESHOLD=os.environ.get('SILENCE_THRESHOLD')
    SPEECH_SENSITIVITY=os.environ.get('SPEECH_SENSITIVITY')
    cobra=pvcobra.create(access_key=os.environ.get('PORCUPINE_KEY'))
    pa=pyaudio.PyAudio()
    streaming=pa.open(
        input_device_index=1, # match the right input device
        channels=1, # porcupine requires single channel input
        rate=audio.sample_rate,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=audio.frame_length
    )
    print("System initiated...")
    activate_listening=False
    audio_buffer=[]
    blank=True
    last_detection=time.time()
    stt_queue=queue.Queue()
    system=False
    try:
        while True:
            audio_bytes=streaming.read(audio.frame_length,exception_on_overflow=False)
            pcm_ints=struct.unpack_from('h'* audio.frame_length,audio_bytes)

            keywords=audio.process(pcm_ints)
            if system is False:
                if keywords==0: # initiation keyword
                    system=True
                    activate_listening,last_detection,audio_buffer,blank=starting_chat()
                    action_q.put("on")
            else:
                if keywords<3:
                    system=True
                    if keywords==1: # camera initiation
                        action_q.put("photo")
                    elif keywords==2: # returning to chat mode on hearing NAVI-END
                        activate_listening,last_detection,audio_buffer,blank=starting_chat()
                        action_q.put("navi_kill")
                if keywords>=3: # stop conversation and send user input to agent
                    system=False
                    activate_listening=False
                    if not blank and len(audio_buffer)>0:
                        process_stt(audio_buffer,audio.sample_rate,stt_queue)
                        audio_buffer=[]
                        continue
                    if keywords==3:
                        action_q.put("agent")
                    elif keywords==4:
                        action_q.put("navi_zoom")

            if activate_listening:
                audio_buffer.append(np.frombuffer(audio_bytes,dtype=np.int16))
                speech_detection=cobra.process(pcm_ints)
                if speech_detection>float(SPEECH_SENSITIVITY):
                    blank=False
                    last_detection=time.time()
                if time.time()-last_detection>float(SILENCE_THRESHOLD):
                    if not blank:
                        process_stt(audio_buffer,audio.sample_rate,stt_queue)
                        audio_buffer=[]
                        blank=True
                        print("Operation: Speech to Text...")
            try:
                stt_result=stt_queue.get_nowait()
                if stt_result:
                    listen_q.put(stt_result)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("Shutting down system...")
    except Exception as e:
        print(f"Receiving interuption: {e}")
    finally:
        # clear cache occupancy
        streaming.stop_stream()
        streaming.close()
        pa.terminate()
        audio.delete()
        cobra.delete()

def process_stt(buffer,rate,res_q):
    combined=np.concatenate(buffer)
    thread=threading.Thread(target=voice_to_text,args=(combined,rate,res_q))
    thread.start()

def starting_chat():
    activate_listening=True
    last_detection=time.time()
    audio_buffer=[]
    blank=True
    return activate_listening,last_detection,audio_buffer,blank