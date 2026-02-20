from zai import ZhipuAiClient
from openai import OpenAI
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from dotenv import load_dotenv
import AssistantGlasses.Agent.code.config as config
from AssistantGlasses.Agent.code.utils import to_base64,img_to_base64

"""
    glm-4.6v-flash using zai's python-sdk
"""

class ZaiAgent():
    def __init__(self,role="default"):
        role_setting=config.SYSTEM_SETTING[role]
        self.conversation=[
            {
                "role":"system",
                "content":role_setting
            }
        ]
        load_dotenv()
        API_KEY=os.environ.get("ZAI_API_KEY")
        if API_KEY!=None:
            print(f"API_KEY status:{bool(API_KEY)}")
        else:
            print("API_KEY not accessed")
        self.client=ZhipuAiClient(api_key=API_KEY,max_retries=3)
    
    def chat_stream(self,input_flow,img_path=False):
        # loading local images
        if img_path:
            prepared=img_to_base64(input_flow)
            if isinstance(prepared,str):
                print("Image upload: ",bool(prepared))
            else:
                print("Failed to load image...")
            self.conversation.append({"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpg;base64,{prepared}"}}]})
        else:
            # add text input to conversation
            if isinstance(input_flow,str):
                self.conversation.append({"role":"user","content":[{"type":"text","text":input_flow}]})
            # add image input to conversation
            else:
                prepared=to_base64(input_flow)
                self.conversation.append({"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpg;base64,{prepared}"}}]})

        response=self.client.chat.completions.create(
            model=config.MODEL[0],
            messages=self.conversation,
            stream=True,
            thinking={
                "type":"disabled" # set this to "disabled" if speed is too slow
            }
        )
        memory=""
        print("Assistant: ",end="") # disable this line afterwards
        for chunk in response:
            content=chunk.choices[0].delta.content
            if content:
                print(content,end="",flush=True) # disable this line afterwards
                memory+=content
        print() # disable this line afterwards
        self.conversation.append({"role":"assistant","content":memory})
        return self.conversation

"""
    glm-4.6v deployed on siliconflow, using openai's python-sdk
"""

class SiliconflowAgent():
    def __init__(self,role="nekomusume"):
        role_setting=config.SYSTEM_SETTING[role]
        print(role_setting)
        self.conversation=[
            {
                "role":"system",
                "content":role_setting
            }
        ]
        load_dotenv()
        API_KEY=os.environ.get("SILICON_FLOW")
        if API_KEY!=None:
            print(f"API_KEY status:{bool(API_KEY)}")
        else:
            print("API_KEY not accessed")
        self.client=OpenAI(api_key=API_KEY,base_url="https://api.siliconflow.cn/v1")

    def chat_stream(self,input_flow,img_path=False):
        # loading local images
        if img_path:
            prepared=img_to_base64(input_flow)
            if isinstance(prepared,str):
                print("Image upload: ",bool(prepared))
            else:
                print("Failed to load image...")
            self.conversation.append({"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpg;base64,{prepared}"}}]})
        else:
            # add text input to conversation
            if isinstance(input_flow,str):
                self.conversation.append({"role":"user","content":[{"type":"text","text":input_flow}]})
            # add image input to conversation
            else:
                prepared=to_base64(input_flow)
                self.conversation.append({"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/jpg;base64,{prepared}"}}]})
        response=self.client.chat.completions.create(
            model=config.MODEL[3],
            messages=self.conversation,
            stream=True
        )
        memory=""
        print("Assistant: ",end="") # disable this line afterwards
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