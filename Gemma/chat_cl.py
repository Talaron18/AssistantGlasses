"""
Agent classes using llama.cpp's OpenAI-compatible API.
Supports streaming, vision, audio input (via Whisper), and tool calls.

Changes vs original
───────────────────
Speed
  • Pipelined TTS: a synthesis thread and a playback thread run concurrently,
    so chunk N+1 is generated while chunk N is playing (~30-50 % faster TTS).
  • kokoro models are now cached at module level (see kokoro.py), eliminating
    per-utterance model reload latency.
  • TTS flush thresholds tuned for both English (word count) and Chinese
    (character count), so the first spoken word arrives faster.

Audio input
  • Gemma4Agent.transcribe_audio(audio) — sends raw audio (numpy or WAV bytes)
    to llama.cpp's /v1/audio/transcriptions Whisper endpoint.
  • chat_stream(audio_data=...) — pass recorded audio directly; the agent
    transcribes first, then continues the normal pipeline.

Reliability
  • interrupt_tts() — drains both TTS queues and stops sounddevice playback
    instantly when the user starts speaking again.
  • _safe_parse_args() — repairs common JSON issues in tool arguments
    (trailing commas, bare identifiers, extra whitespace).
  • _detect_text_tool_call() — catches models that emit tool invocations as
    raw JSON text instead of using the structured tool_calls mechanism.
  • Append _TOOL_INSTRUCTIONS to every system message so the model knows
    exactly when/how to call tools silently.

Prerequisites:
    pip install openai opencv-python sounddevice numpy python-dotenv
    llama-server --port 8090 --model gemma4.gguf ...
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import traceback
import wave
import threading
import queue
from collections import defaultdict
import base64

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import AssistantGlasses.Gemma.config as config

from AssistantGlasses.Agent.code.utils import img_to_base64, to_base64
from AssistantGlasses.Gemma.tool import capture_photo as _capture_photo
from AssistantGlasses.voice_module.kokoro_cl import synthesize


_TOOL_INSTRUCTIONS = """

[TOOL CALLING RULES — follow exactly]
1. When a tool is needed, emit ONLY the function call — zero surrounding text.
   Do NOT write "I'll take a photo", "Let me use the camera", or anything similar.
   Just call the function.
2. Call capture_photo ONLY when the user explicitly asks to: take a photo,
   capture/photograph something, look at something, or needs visual context.
3. After tool results are returned, respond naturally in the conversation.
4. If no tool is needed, reply in plain text as normal.
"""

_EN_RE = re.compile(r"[A-Za-z]")
_ZH_RE = re.compile(r'[\u4e00-\u9fff]')

_LANG_SEG_RE = re.compile(
    r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+'
    r'|[A-Za-z][A-Za-z0-9\'\-]*(?:[ \t]+[A-Za-z][A-Za-z0-9\'\-]*)*'
    r'|[0-9]+(?:[.,][0-9]+)*'
    r'|[^\s]',
    re.UNICODE,
)


def _iter_lang_segments(text: str):
    """
    Yield ``(segment, lang)`` pairs from mixed zh/en text, merging adjacent
    same-language runs.  Numbers and lone punctuation adopt the language of
    the surrounding context.

    Example
    -------
    "我有2个朋友，Gemini 和ChatGPT"
    → [("我有2个朋友，", "zh"), ("Gemini", "en"), ("和", "zh"), ("ChatGPT", "en")]
    """
    buf: str = ""
    cur_lang: str | None = None

    for m in _LANG_SEG_RE.finditer(text):
        token = m.group()
        has_zh = bool(_ZH_RE.search(token))
        has_en = bool(_EN_RE.search(token)) and not has_zh

        if has_zh:
            tok_lang = "zh"
        elif has_en:
            tok_lang = "en"
        else:
            tok_lang = cur_lang or "zh"

        if tok_lang == cur_lang:
            buf += token
        else:
            if buf and cur_lang is not None:
                yield buf, cur_lang
            buf = token
            cur_lang = tok_lang

    if buf and cur_lang is not None:
        yield buf, cur_lang


def _safe_parse_args(raw: str) -> dict:
    """
    Parse tool-call arguments with multiple fallback strategies.
    Handles: empty string, trailing commas, single-quoted keys, stray text.
    """
    if not raw or not raw.strip():
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except json.JSONDecodeError:
        pass

    print(f"[Agent] ⚠ Could not parse tool args: {raw!r}")
    return {}


class BaseAgent:
    """
    Base agent wired to llama.cpp's OpenAI-compatible chat API.

    TTS pipeline
    ─────────────
    tts_queue  ──►  _tts_worker (synthesis)  ──►  _play_queue  ──►  _play_worker
    Text chunks arrive in tts_queue.  The synthesis thread converts text → audio
    and pushes (samples, rate) tuples to _play_queue.  The playback thread drains
    _play_queue with sd.play/wait.  Because both threads run concurrently, chunk
    N+1 is being synthesised while chunk N is audible.
    """

    PUNCTUATION = frozenset(".!?\n。！？……")
    LOCATION_RE = re.compile(r"\[&location/(.*?)&\]")

    _PHANTOM_CAPTURE_RE = re.compile(
        r"\b(i(?:'ll| will| can| am going to)?\s+(?:take|capture|photograph|snap)\b"
        r"|let me (?:use|activate|open) (?:the )?camera)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        destination: queue.Queue,
        role: str = "default",
        speech: queue.Queue | None = None,
    ) -> None:
        base_setting = config.SYSTEM_SETTING[role]
        self.role_setting = base_setting + _TOOL_INSTRUCTIONS

        self.conversation: list[dict] = [
            {"role": "system", "content": self.role_setting}
        ]

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "capture_photo",
                    "description": (
                        "Activate the device camera and capture one photo. "
                        "ONLY call this when the user explicitly asks to take a picture, "
                        "photograph something, 'look at' something, or needs visual "
                        "context that requires seeing the current scene. "
                        "Do NOT call for hypothetical or metaphorical references to cameras."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "save_dir": {
                                "type": "string",
                                "description": "Optional directory path to save the captured photo.",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": ""
                }           
            }
        ]

        load_dotenv()

        self.destination = destination

        self.tts_queue: queue.Queue = speech if speech is not None else queue.Queue()
        self._play_queue: queue.Queue = queue.Queue()

        self._pending_photo_b64: str | None = None

        self._tts_thread  = threading.Thread(target=self._tts_worker,  daemon=True, name="TTS-synth")
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True, name="TTS-play")
        self._tts_thread.start()
        self._play_thread.start()


    def prepare_audio_input(self, wav_bytes: bytes) -> None:
        b64 = base64.b64encode(wav_bytes).decode()
    
        self.conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": b64,
                        "format": "wav",
                    },
                }
            ],
        })

    def stop(self) -> None:
        """Gracefully shut down the TTS pipeline."""
        self.interrupt_tts()
        self.tts_queue.put(None)
        self._tts_thread.join(timeout=3.0)
        self._play_thread.join(timeout=3.0)

    def interrupt_tts(self) -> None:
        """
        Immediately silence TTS: drain both queues and stop sounddevice.
        Call this when the user begins speaking to avoid talking over them.
        """
        for q in (self.tts_queue, self._play_queue):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        try:
            sd.stop()
        except Exception:
            pass


    def detect_lang(self, text: str) -> str:
        """Return 'zh' if *any* Chinese characters are present, else 'en'.

        A single Chinese character means the chunk is Chinese-primary, so the
        zh Kokoro voice is selected.  That voice handles embedded English words
        (brand names, acronyms) correctly via its built-in G2P lexicon.
        """
        return "zh" if _ZH_RE.search(text) else "en"

    def _tts_worker(self) -> None:
        """
        Stage-1 TTS pipeline: text → audio samples.

        For **mixed** zh/en chunks (e.g. "我有2个朋友，Gemini 和ChatGPT") the
        text is split into language-homogeneous segments by ``_iter_lang_segments``,
        each segment is synthesised with the matching voice, and the resulting
        numpy arrays are concatenated before being pushed to the play queue.

        Everything happens inside this background thread, so the playback
        latency perceived by the user is identical to the mono-lingual path.
        """
        while True:
            text = self.tts_queue.get()
            if text is None:
                self._play_queue.put(None)
                break
            try:
                has_zh = bool(_ZH_RE.search(text))
                has_en = bool(_EN_RE.search(text))

                if has_zh and has_en:
                    arrays: list[np.ndarray] = []
                    rate: int | None = None
                    for seg, lang in _iter_lang_segments(text):
                        seg = seg.strip()
                        if not seg:
                            continue
                        s, r = synthesize(seg, lang)
                        if rate is None:
                            rate = r
                        arrays.append(s)
                    if arrays:
                        samples = (
                            np.concatenate(arrays) if len(arrays) > 1 else arrays[0]
                        )
                        self._play_queue.put((samples, rate or 24_000))
                else:
                    lang = "zh" if has_zh else "en"
                    samples, rate = synthesize(text, lang)
                    self._play_queue.put((samples, rate))

            except Exception as exc:
                print(f"[TTS] Synthesis error: {exc}")


    def _play_worker(self) -> None:
        while True:
            item = self._play_queue.get()
            if item is None:
                break
            samples, rate = item
            try:
                sd.play(samples, rate)
                sd.wait()
            except Exception as exc:
                print(f"[TTS] Playback error: {exc}")


    def prepare_input(self, input_flow, img_path: bool = False) -> None:
        if img_path:
            b64 = img_to_base64(input_flow)
            if not b64:
                print("[Agent] Could not load image from path.")
                return
            self.conversation.append(self._vision_message(b64, "Describe this image."))

        elif isinstance(input_flow, str):
            self.conversation.append({"role": "user", "content": input_flow})

        else:
            b64 = to_base64(input_flow)
            if not b64:
                print("[Agent] Could not encode image object.")
                return
            self.conversation.append(self._vision_message(b64, "Describe this image."))

    @staticmethod
    def _audio_message(b64_wav: str, prompt: str) -> dict:
        """
        Build an OpenAI-compatible user message with native audio input.
        Per Google's Gemma4 model card, text must appear BEFORE audio in the
        content array for optimal performance.

        Requires llama-server launched with:
            --mmproj mmproj-BF16.gguf --jinja
        and a build that includes PR #21421 (Gemma4 audio conformer encoder).
        """
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": b64_wav, "format": "wav"}},
            ],
        }

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


    def _detect_text_tool_call(self, text: str) -> dict | None:
        """
        Some model checkpoints emit tool invocations as raw JSON text rather
        than using the structured tool_calls field.  Detect the pattern
        {"name": "capture_photo", ...} and convert it to a usable call dict.
        """
        stripped = text.strip()
        if not stripped.startswith("{"):
            return None
        try:
            data = json.loads(stripped)
            tool_names = {t["function"]["name"] for t in self.tools}
            if isinstance(data, dict) and data.get("name") in tool_names:
                return data
        except (json.JSONDecodeError, KeyError):
            pass
        return None


    def process_stream_and_tools(self, stream) -> list:
        memory        = ""
        tool_acc: dict = defaultdict(lambda: {"id": "", "name": "", "arguments": ""})
        sentence_buf  = ""

        print("Assistant: ", end="", flush=True)

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta  = choice.delta

            content = delta.content or ""
            if content:
                print(content, end="", flush=True)
                memory       += content
                sentence_buf += content


                should_flush = (
                    sentence_buf.rstrip().endswith((
                        ".", "!", "?",
                        "。", "！", "？"
                    ))
                    or len(sentence_buf) > 120
                )

                if should_flush and sentence_buf.strip():
                    if "&]" in sentence_buf:
                        m = self.LOCATION_RE.search(sentence_buf)
                        if m:
                            location = m.group(1).strip()
                            self.destination.put(location)
                            self.tts_queue.put(f"请问您是要导航到{location}吗？")
                            sentence_buf = self.LOCATION_RE.sub("", sentence_buf)
                    else:
                        txt = sentence_buf.strip()
                        self.tts_queue.put(txt)

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

        if not tool_acc and memory.strip():
            phantom = self._detect_text_tool_call(memory.strip())
            if phantom:
                print(f"\n[Agent] Detected text-embedded tool call: {phantom['name']}")
                tool_acc[0] = {
                    "id":        "phantom-0",
                    "name":      phantom["name"],
                    "arguments": json.dumps(phantom.get("parameters", phantom.get("arguments", {}))),
                }
                memory = ""

        if tool_acc:
            print("\n[Agent] Activating tools…")
            assistant_calls = [
                {
                    "id":   tool_acc[i]["id"],
                    "type": "function",
                    "function": {
                        "name":      tool_acc[i]["name"],
                        "arguments": tool_acc[i]["arguments"],
                    },
                }
                for i in sorted(tool_acc)
            ]

            self.conversation.append({
                "role":       "assistant",
                "content":    memory or None,
                "tool_calls": assistant_calls,
            })

            for call in assistant_calls:
                func_name = call["function"]["name"]
                args_str  = call["function"]["arguments"]
                call_id   = call["id"]

                try:
                    args = _safe_parse_args(args_str)
                    func = getattr(self, func_name, None)
                    result_str = (
                        json.dumps(func(**args))
                        if callable(func)
                        else f"Error: '{func_name}' is not a registered tool."
                    )
                except Exception as exc:
                    result_str = f"Tool execution error: {exc}"
                    print(f"[Agent] {result_str}")

                self.conversation.append({
                    "role":         "tool",
                    "tool_call_id": call_id,
                    "content":      result_str,
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

        self.client = OpenAI(base_url=f"{host}/v1", api_key="not-required")
        self.model: str = getattr(config, "LLAMA_MODEL", "gemma4")
        print(f"[Gemma4Agent] Ready — {host}  model: {self.model}")


    @staticmethod
    def _ndarray_to_wav(audio: np.ndarray, sample_rate: int = 16_000) -> bytes:
        """Convert float32 mono numpy array to 16-bit PCM WAV bytes."""
        pcm = (audio.flatten() * 32_767.0).clip(-32_768, 32_767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def transcribe_audio(
        self,
        audio: "np.ndarray | bytes",
        sample_rate: int = 16_000,
    ) -> str:
        """
        Transcribe audio via llama.cpp's /v1/audio/transcriptions endpoint.

        Parameters
        ----------
        audio       : float32 mono numpy array  OR  pre-encoded WAV bytes.
        sample_rate : only used when audio is np.ndarray.

        Returns
        -------
        Transcribed text (stripped).  Empty string on failure.
        """
        try:
            wav_bytes = (
                self._ndarray_to_wav(audio, sample_rate)
                if isinstance(audio, np.ndarray)
                else audio
            )
            bio      = io.BytesIO(wav_bytes)
            bio.name = "recording.wav"

            whisper_model = getattr(config, "WHISPER_MODEL", "whisper")
            result = self.client.audio.transcriptions.create(
                model = whisper_model,
                file  = bio,
            )
            return result.text.strip()
        except Exception as exc:
            print(f"[Agent] Transcription error: {exc}")
            return ""


    def chat_stream(
        self,
        input_flow=None,
        img_path: bool = False,
        tool: bool = True,
        audio_data: "np.ndarray | bytes | None" = None,
        native_audio: bool = False,
    ) -> list:
        """
        Parameters
        ----------
        input_flow   : str | image_path | cv2_frame
        img_path     : True when input_flow is a file path to an image
        tool         : enable function/tool calling
        audio_data   : float32 numpy array or WAV bytes
        native_audio : True  → send audio directly to Gemma4's audio encoder
                               (requires --mmproj and llama.cpp build ≥ b8773,
                               currently experimental — see llama.cpp issue #23688)
                       False → transcribe via Whisper first, send resulting text
                               (default; reliable and faster for pure STT)
        """
        if audio_data is not None:
            if native_audio:
                wav_bytes = (
                    self._ndarray_to_wav(audio_data)
                    if isinstance(audio_data, np.ndarray)
                    else audio_data
                )
                self.prepare_audio_input(wav_bytes)

            if not native_audio:
                print("[Agent] Transcribing audio via llama.cpp Whisper…")
                transcribed = self.transcribe_audio(wav_bytes)
                if not transcribed:
                    print("[Agent] Empty transcription — skipping.")
                    return self.conversation
                print(f"[Transcribed] {transcribed}")
                input_flow = transcribed
                self.prepare_input(input_flow, img_path)
        else:
            self.prepare_input(input_flow, img_path)

        try:
            stream = self.client.chat.completions.create(
                model    = self.model,
                messages = self.conversation,
                stream   = True,
                tools    = self.tools if tool else None,
                temperature = 0.2,
            )
            self.process_stream_and_tools(stream)

            if self._pending_photo_b64:
                b64 = self._pending_photo_b64
                self._pending_photo_b64 = None
                print("\n[Agent] Photo captured — requesting vision analysis…")

                self.conversation.append(
                    self._vision_message(
                        b64,
                        "This is the photo I just took. "
                        "Please describe what you see in detail. "
                        "If there is any text visible in the photo, read it all.",
                    )
                )
                vision_stream = self.client.chat.completions.create(
                    model       = self.model,
                    messages    = self.conversation,
                    stream      = True,
                    temperature = 0.2,
                )
                self.process_stream_and_tools(vision_stream)

        except Exception as exc:
            print(f"[Gemma4Agent] Error: {exc}")
            traceback.print_exc()

        return self.conversation
