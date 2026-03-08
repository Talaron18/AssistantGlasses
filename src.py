import os
import sys
import threading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AssistantGlasses.Agent.code.chat import SiliconflowAgent
from AssistantGlasses.speech_module.stream.record import stream
import queue

class Control():
    def __init__(self):
        self.speech=queue.Queue()
        self.listen=queue.Queue()
        self.action=queue.Queue()
        self.agent=SiliconflowAgent(speech=self.speech)
        
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
                current_action=self.action.get()
                if current_action=="agent":
                    print("Sending request...")
                    self.speech.put("请稍等，正在连接服务器。")
                    user_input=self.listen.get()
                    if user_input:
                        print(user_input)
                        self.agent.chat_stream(user_input,tool=False)
                elif current_action=="photo":
                    # put the camera-control function here
                    print("Taking picuture...")
                elif current_action=="navi_zoom":
                    # put the navigation initialization function here
                    print("Starting navigation...")
                elif current_action=="navi_kill":
                    # set the status of navigation to False
                    print("Navigation palsed...")
                elif current_action=="on":
                    print("On your command...")
            except queue.Empty:
                print("...\n")
            except KeyboardInterrupt:
                print("Shutting down conversation...")
                break

if __name__=="__main__":
    run=Control()
    run.listen_to_input()