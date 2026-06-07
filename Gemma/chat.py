import os
import re
import sys
import json
import traceback
import threading
import queue
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

# 确保路径适配你的项目结构
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import AssistantGlasses.Gemma.config as config

from AssistantGlasses.Agent.code.utils import to_base64, img_to_base64
from AssistantGlasses.Gemma.tool import capture_photo as _capture_photo

# 引入优化后的 TTS 模块与新写的 Mic 模块
from AssistantGlasses.voice_module.kokoro import speak, init_tts
from AssistantGlasses.speech_module.mic import PushToTalkRecorder

# ---------------------------------------------------------------------------
# BaseAgent
# ---------------------------------------------------------------------------

class BaseAgent:
    PUNCTUATION = frozenset("...\n……")
    LOCATION_RE = re.compile(r"\[&location/(.*?)&\]")

    def __init__(
        self,
        destination: queue.Queue,
        lan: str = "default",
        role: str = "default",
        speech: queue.Queue | None = None,
    ) -> None:
        # 如果 config 中没有该角色，默认提供一个系统提示
        self.role_setting = getattr(config, "SYSTEM_SETTING", {}).get(role, "You are a helpful voice assistant.")
        self.conversation: list[dict] = [
            {"role": "system", "content": self.role_setting}
        ]

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "capture_photo",
                    "description": "Activate the default system camera and capture a single photo.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "save_dir": {"type": "string"}
                        },
                        "required": [],
                    },
                },
            }
        ]

        load_dotenv()

        self.destination = destination
        self.lan = lan
        self.tts_queue: queue.Queue = speech if speech is not None else queue.Queue()
        self.input_mode = "text"

        self._pending_photo_b64: str | None = None

        # 启动 TTS 消费者线程
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

    def stop(self) -> None:
        self.tts_queue.put(None)
        self._tts_thread.join()
    
    def _tts_worker(self) -> None:
        # 在子线程中提前初始化 TTS，防止主线程卡顿
        init_tts(self.lan)
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            try:
                speak(text, self.lan)
            except Exception as exc:
                print(f"[TTS] Error: {exc}")
            finally:
                self.tts_queue.task_done()

    def set_input_mode(self, mode: str):
        assert mode in ("audio", "text")
        self.input_mode = mode
        print(f"[Agent] Input mode switched to: {mode.upper()}")

    def prepare_input(self, input_flow, img_path: bool = False) -> None:
        if img_path:
            b64 = img_to_base64(input_flow)
            if b64:
                self.conversation.append(self._vision_message(b64, "Describe this image."))
        elif isinstance(input_flow, str):
            self.conversation.append({"role": "user", "content": input_flow})
        else:
            b64 = to_base64(input_flow)
            if b64:
                self.conversation.append(self._vision_message(b64, "Describe this image."))

    def prepare_audio_input(self, audio_b64: str) -> None:
        self.conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    },
                }
            ],
        })

    @staticmethod
    def _vision_message(b64: str, prompt: str) -> dict:
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }

    def capture_photo(self, save_dir: str = "./photos") -> dict:
        result = _capture_photo(save_dir=save_dir)
        if result.get("success") and result.get("base64"):
            self._pending_photo_b64 = result.pop("base64")
        return result

    def process_stream_and_tools(self, stream) -> list:
        memory = ""
        tool_acc: dict = defaultdict(lambda: {"id": "", "name": "", "arguments": ""})
        sentence_buf = ""

        print("Assistant: ", end="", flush=True)

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            # ---- Text delta ------------------------------------------------
            content = delta.content or ""
            if content:
                print(content, end="", flush=True)
                memory += content
                sentence_buf += content

                should_flush = any(p in sentence_buf for p in self.PUNCTUATION)

                if should_flush and sentence_buf.strip():
                    if "&]" in sentence_buf:
                        match = self.LOCATION_RE.search(sentence_buf)
                        if match:
                            location = match.group(1).strip()
                            self.destination.put(location)
                            self.tts_queue.put(f"请问您是要导航到{location}吗？")
                            # 替换掉导航标签后，如果还有剩余文本，不应该丢弃
                            sentence_buf = self.LOCATION_RE.sub("", sentence_buf)
                    else:
                        # 把完整的一句话放进 TTS 队列
                        self.tts_queue.put(sentence_buf.strip())
                        # 清空缓冲区，开始攒下一句话
                        sentence_buf = ""

            # ---- Tool-call deltas ------------------------------------------
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if tc.id:
                        tool_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_acc[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_acc[idx]["arguments"] += tc.function.arguments

            if choice.finish_reason in ("stop", "tool_calls"):
                break

        print()  # newline after streamed output

        # Flush any remaining sentence fragment (比如大模型最后没加标点就结束了)
        if sentence_buf.strip():
            self.tts_queue.put(sentence_buf.strip())

        # ---- Execute tool calls -------------------------------------------
        if tool_acc:
            print("\n[Agent] Activating tools...")
            assistant_calls = [
                {
                    "id": tool_acc[i]["id"],
                    "type": "function",
                    "function": {
                        "name": tool_acc[i]["name"],
                        "arguments": tool_acc[i]["arguments"],
                    },
                }
                for i in sorted(tool_acc)
            ]

            self.conversation.append({
                "role": "assistant",
                "content": memory or None,
                "tool_calls": assistant_calls,
            })

            for call in assistant_calls:
                func_name = call["function"]["name"]
                args_str  = call["function"]["arguments"]
                call_id   = call["id"]

                try:
                    args = json.loads(args_str) if args_str else {}
                    func = getattr(self, func_name, None)
                    if callable(func):
                        result_str = json.dumps(func(**args))
                    else:
                        result_str = f"Error: '{func_name}' is not a registered tool."
                except Exception as exc:
                    result_str = f"Tool execution error: {exc}"
                    print(f"[Agent] {result_str}")

                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_str,
                })
        else:
            self.conversation.append({"role": "assistant", "content": memory})

        return self.conversation


# ---------------------------------------------------------------------------
# Gemma4Agent
# ---------------------------------------------------------------------------

class Gemma4Agent(BaseAgent):
    def __init__(
        self,
        destination: queue.Queue,
        role: str = "default",
        speech: queue.Queue | None = None,
        host: str = "http://localhost:8090",
    ) -> None:
        super().__init__(destination=destination, role=role, speech=speech)

        self.client = OpenAI(
            base_url=f"{host}/v1",
            api_key="not-required",
        )
        self.model: str = getattr(config, "LLAMA_MODEL", "gemma4")
        print(f"[Gemma4Agent] Ready — {host}  model: {self.model}")

    def chat_stream(
        self,
        input_flow,
        img_path: bool = False,
        tool: bool = True,
    ) -> list:
        # 自动根据当前的模式决定路由逻辑
        if self.input_mode == "audio":
            self.prepare_audio_input(input_flow)
        else:
            self.prepare_input(input_flow, img_path)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                stream=True,
                tools=self.tools if tool else None,
            )
            self.process_stream_and_tools(stream)

            if self._pending_photo_b64:
                b64 = self._pending_photo_b64
                self._pending_photo_b64 = None
                print("\n[Agent] Photo captured — requesting vision analysis...")

                self.conversation.append(
                    self._vision_message(
                        b64,
                        "This is the photo I just took. Please describe what you see in detail. If you detect any text in the photo, read them all."
                    )
                )

                vision_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation,
                    stream=True,
                )
                self.process_stream_and_tools(vision_stream)

        except Exception as exc:
            print(f"[Gemma4Agent] Error: {exc}")
            traceback.print_exc()

        return self.conversation


# ---------------------------------------------------------------------------
# Interactive CLI Loop (可用于独立测试运行)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dummy_dest = queue.Queue()
    agent = Gemma4Agent(destination=dummy_dest, role="default")
    recorder = PushToTalkRecorder()
    audio_generator = recorder.listen()

    print("\n--- Assistant Ready ---")
    print("Commands: /audio (Switch to voice), /text (Switch to text typing), /quit")
    
    # 默认开启 Text 模式测试
    agent.set_input_mode("text")

    try:
        while True:
            if agent.input_mode == "text":
                cmd = input("\n> ")
                if cmd.strip() == "": continue
                if cmd == "/quit": break
                if cmd == "/audio":
                    agent.set_input_mode("audio")
                    continue
                if cmd == "/text":
                    print("[Agent] Already in text mode.")
                    continue
                
                agent.chat_stream(cmd)

            else:
                cmd = input("\n[AUDIO MODE] Type /text to switch back, /quit to exit. Or press ENTER to ready mic: ")
                if cmd == "/quit": break
                if cmd == "/text":
                    agent.set_input_mode("text")
                    continue
                
                # 开始监听空格键录音
                audio_data = next(audio_generator)
                if audio_data and "base64" in audio_data:
                    agent.chat_stream(audio_data["base64"])
                    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        agent.stop()