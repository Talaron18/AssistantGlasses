import os
import sys
import threading
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AssistantGlasses.Agent.code.chat import SiliconflowAgent
from AssistantGlasses.speech_module.stream.record import stream
from AssistantGlasses.navigation_module.core.nav_controller import NavController
import queue

"""
    yet to be tested
"""
class Control():
    def __init__(self):
        self.speech=queue.Queue()
        self.listen=queue.Queue()
        self.action=queue.Queue()
        self.destination=queue.Queue()
        self.agent=SiliconflowAgent(destination=self.destination,speech=self.speech)
        self.nav_thread=NavController(
            tts_queue=self.speech,
            nav_queue=self.destination
        )
        self.nav_thread.daemon=True
        self.nav_thread.start()

    def listen_to_input(self):
        stream_thread=threading.Thread(
            target=stream,
            args=(self.listen,self.action),
            daemon=True
        )
        stream_thread.start()
        print("Ready for inputs...")
        while True:
            try:
                try:
                    current_action=self.action.get(timeout=0.1)
                except queue.Empty:
                    continue
                if current_action=="agent":
                    print("Sending request...")
                    self.speech.put("请稍等，正在连接服务器。")
                    user_input=""
                    stt_time=time.time()
                    try:
                        user_input+=self.listen.get()
                    except queue.Empty:
                        print("STT timed out or returned no text...")
                        continue
                    while not self.listen.empty():
                        user_input+=" "+self.listen.get()
                    stt_end=time.time()
                    print(f"-->stt took {stt_end-stt_time:.2f} seconds")
                    if user_input.strip():
                        print(user_input)
                        agent_start=time.time()
                        self.agent.chat_stream(user_input,tool=False)
                        print(f"--> Agent response time: {time.time()-agent_start:.2f} seconds")
                        
                elif current_action=="photo":
                    # put the camera-control function here
                    print("Taking picuture...")
                elif current_action=="navi_zoom":
                    self.destination.put("良乡大学城西地铁站")
                    # put the navigation initialization function here
                    print("Starting navigation...")
                elif current_action=="navi_kill":
                    # set the status of navigation to False
                    print("Navigation palsed...")
                    self.nav_thread.shutdown()
                    self.destination.put("STOP")
                elif current_action=="on":
                    print("On your command...")
            
            except KeyboardInterrupt:
                print("Shutting down conversation...")
                break

if __name__=="__main__":
    run=Control()
    run.listen_to_input()