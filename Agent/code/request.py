from zai import ZhipuAiClient
from openai import OpenAI
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from dotenv import load_dotenv
import AssistantGlasses.Agent.code.config as config
from AssistantGlasses.Agent.code.utils import to_base64,img_to_base64
from AssistantGlasses.Agent.test.tool_test import quicktest

"""
    glm-4.6v-flash using zai's python-sdk
"""

class ZaiAgent():
    def __init__(self,role="default"):
        role_setting=config.SYSTEM_SETTING[role]
        self.quicktest=quicktest
        self.tools=[
            {
                "type":"function",
                "name":"quicktest", # enter function name
                "description":"a quick test for external tools",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "mode":{
                            "type":"string",
                            "description":"status e.g. ON, OFF"
                        }
                    },
                    "required":["mode"]
                }
            }
        ]
        self.conversation=[
            {
                "role":"system",
                "content":role_setting
            }
        ]
        load_dotenv()
        API_KEY=os.environ.get("ZAI_API_KEY")
        if API_KEY is not None:
            print(f"API_KEY status:{bool(API_KEY)}")
        else:
            print("API_KEY not accessed")
        self.client=ZhipuAiClient(api_key=API_KEY,max_retries=3)
    
    def chat_stream(self,input_flow,img_path=False,tool=True):
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
            tools=self.tools if tool else None,
            thinking={
                "type":"disabled" # set this to "disabled" if speed is too slow
            }
        )
        memory=""
        tool_id=None
        func1_name=""
        func1_args=""
        print("Assistant: ",end="",flush=True) # disable this line afterwards
        for chunk in response:      
            delta=chunk.choices[0].delta
            if delta.content:
                print(delta.content,end="",flush=True) # disable this line afterwards
                memory+=delta.content
            if delta.tool_calls:
                print("Activating tools...\n")
                if delta.tool_calls[0].id:
                    tool_id=delta.tool_calls[0].id
                tool_func=delta.tool_calls[0].function
                if tool_func.name:
                    func1_name+=tool_func.name
                if tool_func.arguments:
                    func1_args+=tool_func.arguments
        print() # disable this line afterwards
        if tool_id:
            import json
            args=json.loads(func1_args)
            func=getattr(self,func1_name,None)
            if callable(func):
                result=func(**args)
                self.conversation.append({
                    "role":"assistant",
                    "tool_calls":[{
                        "id":tool_id,
                        "type":"function",
                        "function":{"name":func1_name,"arguments":func1_args}
                    }]
                })
                self.conversation.append({
                    "role":"tool",
                    "tool_call_id":tool_id,
                    "content":str(result)
                })
        else:
            self.conversation.append({"role":"assistant","content":memory})
        return self.conversation

"""
    glm-4.6v deployed on siliconflow, using openai's python-sdk
"""

class SiliconflowAgent():
    def __init__(self,role="default"):
        role_setting=config.SYSTEM_SETTING[role]
        # print(role_setting)
        self.quicktest=quicktest
        self.tools=[
            {
                "type":"function",
                "function":{
                    "name":"quicktest", # enter function name
                    "description":"play some music",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "mode":{
                                "type":"string",
                                "enum":["ON","OFF"],
                                "description":"operation mode of the tool"
                            }
                        },
                        "required":["mode"]
                    }
                }
            }
        ]
        self.conversation=[
            {
                "role":"system",
                "content":role_setting
            }
        ]
        load_dotenv()
        API_KEY=os.environ.get("SILICON_FLOW")
        if API_KEY is not None:
            print(f"API_KEY status:{bool(API_KEY)}")
        else:
            print("API_KEY not accessed")
        self.client=OpenAI(api_key=API_KEY,base_url="https://api.siliconflow.cn/v1")

    def chat_stream(self,input_flow,img_path=False,tool=True):
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
        try:
            response=self.client.chat.completions.create(
                model=config.MODEL[3],
                messages=self.conversation,
                tools=self.tools if tool else None,
                tool_choice="auto",
                stream=True,
                #extra_body={
                #    "thinking_budget":4096 # change this parameter to adjust response time, token consumption and accuracy
                #}
            )
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        memory=""
        tool_id=None
        func1_name=""
        func1_args=""
        print("Assistant: ",end="",flush=True) # disable this line afterwards
        for chunk in response:
            if not chunk.choices:
                continue
            delta=chunk.choices[0].delta
            if hasattr(delta,"content") and delta.content:
                print(delta.content,end="",flush=True) # disable this line afterwards
                memory+=delta.content
            else:
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    print(delta.reasoning_content,end="",flush=True)
                    memory+=delta.reasoning_content
            if hasattr(delta,"tool_calls") and delta.tool_calls:
                print("Activating tools...")
                if delta.tool_calls[0].id:
                    tool_id=delta.tool_calls[0].id
                if delta.tool_calls[0].function:
                    tool_func=delta.tool_calls[0].function
                    if tool_func.name:
                        func1_name+=tool_func.name
                    if tool_func.arguments:
                        func1_args+=tool_func.arguments
        print() # disable this line afterwards
        if tool_id:
            print(f"System: Tool {func1_name} triggered...")
            import json
            try:
                args=json.loads(func1_args)
                func=getattr(self,func1_name,None)
                if callable(func):
                    result=func(**args)
                    self.conversation.append({
                        "role":"assistant",
                        "content":None,
                        "tool_calls":[{
                            "id":tool_id,
                            "type":"function",
                            "function":{"name":func1_name,"arguments":func1_args}
                        }]
                    })
                    self.conversation.append({
                        "role":"tool",
                        "tool_call_id":tool_id,
                        "content":str(result) if result is not None else "success"
                    })
            except json.JSONDecodeError:
                print("Invalid JSON for tool arguments...")
        else:
            self.conversation.append({"role":"assistant","content":memory})
        return self.conversation

"""
    the wake-up words shouldn't be sent to agent
    make sure this problem is solved
"""