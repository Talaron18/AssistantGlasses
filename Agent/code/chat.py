"""
     A polished version of request functions modified with Gemini,
     removing redundant logic and adding wake word stripping
"""

import os
import sys
import json
import traceback
import threading
import queue
from dotenv import load_dotenv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import AssistantGlasses.Agent.code.config as config
from AssistantGlasses.Agent.code.utils import to_base64, img_to_base64
from AssistantGlasses.Agent.test.tool_test import quicktest
from zai import ZhipuAiClient
from openai import OpenAI
from AssistantGlasses.voice_module.read import TTS

"""
    base class handling both zai and siliconflow models
"""
class BaseAgent:
    def __init__(self, role="default"):
        self.role_setting = config.SYSTEM_SETTING[role]
        self.quicktest = quicktest
        self.conversation = [
            {"role": "system", "content": self.role_setting}
        ]
        self.tools = [{
            "type": "function",
            "function": {
                "name": "quicktest",
                "description": "a quick test for external tools",
                "parameters": {
                    "type": "object",
                    "properties": {"mode": {"type": "string", "enum": ["ON", "OFF"], "description": "operation mode of the tool"}},
                    "required": ["mode"]
                }
            }
        }]
        load_dotenv()
        self.tts_queue=queue.Queue()
        self.tts_thread=threading.Thread(target=self.tts_go,daemon=True)
        self.tts_thread.start()

    # remove wake words from text input
    def strip_wake_words(self, text: str) -> str:
        WAKE_WORDS=config.WAKE_WORDS
        lower_text = text.lower().strip()
        for wake_word in WAKE_WORDS:
            if lower_text.startswith(wake_word):
                return text[len(wake_word):].lstrip(" ,.!?:")
        return text
    
    def tts_go(self):
        while True:
            text=self.tts_queue.get()
            if text is None:
                break
            try:
                tts=TTS()
                tts.speak(text)
                del(tts)
            except Exception as e:
                print(f"TTS error: {e}")
            finally:
                self.tts_queue.task_done()

    def prepare_input(self, input_flow, img_path=False):
        # handle image path for image uploads
        if img_path:
            prepared = img_to_base64(input_flow)
            print("Image upload: ", bool(prepared))
            if not prepared:
                print("Failed to load image...")
                return
            self.conversation.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{prepared}"}}]})
        else:
            if isinstance(input_flow, str):
                # clean wake words before adding to conversation
                cleaned_input = self.strip_wake_words(input_flow)
                self.conversation.append({"role": "user", "content": [{"type": "text", "text": cleaned_input}]})
            else:
                # handle cv2 and PIL images for uploads
                prepared = to_base64(input_flow)
                self.conversation.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpg;base64,{prepared}"}}]})

    def process_stream_and_tools(self, response):
        """Shared logic to stream responses and execute tool calls."""
        memory = ""
        tool_id = None
        func_name = ""
        func_args = ""
        sentence_buffer="" 
        punctuation=['.','!','?','\n','。','！','？','……']
        print("Assistant: ", end="", flush=True)
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            
            # Stream text content
            content=getattr(delta,"content",None) or getattr(delta,"reasoning_content",None)
            if content:
                print(content,end="",flush=True)
                memory+=content
                sentence_buffer+=content
                if any(punct in sentence_buffer for punct in punctuation):
                    self.tts_queue.put(sentence_buffer.strip())
                    sentence_buffer=""

            # Stream tool calls
            if getattr(delta, "tool_calls", None):
                if not tool_id:
                    print("\nActivating tools...\n")
                
                tool_call = delta.tool_calls[0]
                if getattr(tool_call, "id", None):
                    tool_id = tool_call.id
                
                tool_func = getattr(tool_call, "function", None)
                if tool_func:
                    if getattr(tool_func, "name", None):
                        func_name += tool_func.name
                    if getattr(tool_func, "arguments", None):
                        func_args += tool_func.arguments             
        print() # End of stream line break
        if sentence_buffer.strip():
            self.tts_queue.put(sentence_buffer.strip())
        # Execute tool if one was called
        if tool_id:
            try:
                args = json.loads(func_args)
                func = getattr(self, func_name, None)
                if callable(func):
                    result = func(**args)
                    self.conversation.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": func_name, "arguments": func_args}
                        }]
                    })
                    self.conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": str(result) if result is not None else "success"
                    })
            except json.JSONDecodeError:
                print("Invalid JSON for tool arguments...")
        else:
            self.conversation.append({"role": "assistant", "content": memory})
        return self.conversation


class ZaiAgent(BaseAgent):
    """glm-4.6v-flash using zai's python-sdk"""
    def __init__(self, role="default"):
        super().__init__(role)
        api_key = os.environ.get("ZAI_API_KEY")
        print(f"API_KEY status (ZAI): {bool(api_key)}")
        self.client = ZhipuAiClient(api_key=api_key, max_retries=3)
    
    def chat_stream(self, input_flow, img_path=False, tool=True):
        self.prepare_input(input_flow, img_path)
        
        response = self.client.chat.completions.create(
            model=config.MODEL[0],
            messages=self.conversation,
            stream=True,
            tools=self.tools if tool else None,
            thinking={"type": "disabled"}
        )
        return self.process_stream_and_tools(response)


class SiliconflowAgent(BaseAgent):
    """glm-4.6v deployed on siliconflow, using openai's python-sdk"""
    def __init__(self, role="default"):
        super().__init__(role)
        api_key = os.environ.get("SILICON_FLOW")
        print(f"API_KEY status (SiliconFlow): {bool(api_key)}")
        self.client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    def chat_stream(self, input_flow, img_path=False, tool=True):
        self.prepare_input(input_flow, img_path)
        
        try:
            response = self.client.chat.completions.create(
                model=config.MODEL[3],
                messages=self.conversation,
                tools=self.tools if tool else None,
                stream=True,
                extra_body={"thinking_budget": 2048}
            )
            return self.process_stream_and_tools(response)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return self.conversation