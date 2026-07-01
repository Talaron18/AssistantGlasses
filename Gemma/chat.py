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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import AssistantGlasses.Gemma.config as config

from AssistantGlasses.Agent.code.utils import to_base64, img_to_base64
from AssistantGlasses.Gemma.tool import capture_photo as _capture_photo

from AssistantGlasses.voice_module.kokoro import speak, init_tts
from AssistantGlasses.speech_module.mic import PushToTalkRecorder


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

        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

    def stop(self) -> None:
        self.tts_queue.put(None)
        self._tts_thread.join()
    
    def _tts_worker(self) -> None:
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
                            sentence_buf = self.LOCATION_RE.sub("", sentence_buf)
                    else:
                        self.tts_queue.put(sentence_buf.strip())
                        sentence_buf = ""

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

        print()

        if sentence_buf.strip():
            self.tts_queue.put(sentence_buf.strip())

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


if __name__ == "__main__":
    dummy_dest = queue.Queue()
    agent = Gemma4Agent(destination=dummy_dest, role="default")
    recorder = PushToTalkRecorder()
    audio_generator = recorder.listen()

    print("\n--- Assistant Ready ---")
    print("Commands: /audio (Switch to voice), /text (Switch to text typing), /quit")
    
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
                
                audio_data = next(audio_generator)
                if audio_data and "base64" in audio_data:
                    agent.chat_stream(audio_data["base64"])
                    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        agent.stop()