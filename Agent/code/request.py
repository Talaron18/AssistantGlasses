from zai import ZaiClient
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from dotenv import load_dotenv
import AssistantGlasses.Agent.code.config as config
from AssistantGlasses.Agent.code.utils import to_base64

class CoreAgent():
    def __init__(self,role="default"):
        role_setting=config.SYSTEM_SETTING[role]
        self.conversation=[
            {
                "role":"system",
                "content":role_setting
            }
        ]
        load_dotenv()
        API_KEY=os.environ.get("API_KEY")
        self.client=ZaiClient(api_key=API_KEY,max_retries=3)
    
    def chat_stream(self,input_flow):
        # add text input to conversation
        if isinstance(input_flow,str):
            self.conversation.append({"role":"user","content":[{"type":"text","text":input_flow}]})
        # add image input to conversation
        else:
            prepared=to_base64(input_flow)
            self.conversation.append({"role":"user","content":[{"type":"img_url","img_url":{"url":f"data:image/jpg;base64,{prepared}"}}]})

        response=self.client.chat.completions.create(
            model=config.MODEL,
            messages=self.conversation,
            stream=True,
            thinking={
                "type":"disabled" # set this to "disabled" if speed is too slow
            }
        )
        memory=""
        print("Assistant",end="") # disable this line afterwards
        for chunk in response:
            content=chunk.choices[0].delta.content
            if content:
                print(content,end="",flush=True) # disable this line afterwards
                memory+=content
        print() # disable this line afterwards
        self.conversation.append({"role":"assistant","content":memory})
        return self.conversation

"""
    the wake-up words shouldn't be sent to agent
    make sure this problem is solved
"""