import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from AssistantGlasses.Agent.code.request import SiliconflowAgent

if __name__=="__main__":
    begin=input("Start conversation: ")
    try:
        agent=SiliconflowAgent()
        agent.chat_stream(begin)
        path="C:/Users/32873/.vscode/python/AssistantGlasses/Agent/test_image.jpg"
        agent.chat_stream(path,img_path=True)
        while True:
            chat=input("Enter: ")
            agent.chat_stream(chat)
    except Exception as e:
        print("Terminating conversation...")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()