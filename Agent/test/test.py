from dotenv import load_dotenv
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
#from AssistantGlasses.Agent.code.request import SiliconflowAgent
from AssistantGlasses.Agent.code.chat import SiliconflowAgent
if __name__=="__main__":
    begin=input("Start conversation: ")
    load_dotenv()
    try:
        agent=SiliconflowAgent()
        agent.chat_stream(begin)
        path=os.environ.get("IMG")
        
        while True:
            chat=input("Enter: ")
            agent.chat_stream(chat)
            if chat=="img":
                agent.chat_stream(path,img_path=True,tool=False)
            if chat=="Ending conversation":
                break
        #print(agent.conversation)
    except Exception as e:
        print("Terminating conversation...")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()