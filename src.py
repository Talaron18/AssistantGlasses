import os
import sys
import threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AssistantGlasses.Agent.code.chat import SiliconflowAgent
from AssistantGlasses.speech_module.stream.record import stream
import queue
from AssistantGlasses.speech_module.stream.utils import keyword_check
class Control():
    def __init__(self):
        self.speech=queue.Queue()
        self.listen=queue.Queue()
        self.agent=SiliconflowAgent()
        self.threads=[]
        self.agent_thread=threading.Thread(target=self.agent,daemon=True)
        self.threads.append(self.agent_thread)
        self.recognize_thread=threading.Thread(target=stream())
        self.threads.append(self.recognize_thread)
        # open new threads for different modules
        # append modules to self.threads
        for t in self.threads:
            t.start()

    def new_speech(self,text:str):
        self.speech.put(text.strip())

    def listen_to_input(self):
        while True:
            input=self.listen.get()
            self.agent.chat_stream(input)