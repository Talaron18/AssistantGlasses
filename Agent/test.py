import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from AssistantGlasses.Agent.code.request import CoreAgent

if __name__=="__main__":
    input=input("Start conversation: ")
    try:
        agent=CoreAgent()
        while True:
            agent.chat_stream(input)
            input=input("")
    except:
        print("Terminating conversation...")