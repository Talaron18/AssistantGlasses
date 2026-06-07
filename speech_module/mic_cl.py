"""
mic.py — Push-to-talk + text console for Gemma4Agent.

AUDIO mode  Hold SPACE ─► record ─► release ─► transcribe ─► send to Gemma4
TEXT mode   Type + Enter to send

Global commands (work in either mode, type and press Enter):
  /text  or /t    Switch to text input mode
  /audio or /a    Switch to push-to-talk mode
  /quit  or /q    Exit

Interruption
  Pressing SPACE while in audio mode (once the agent has finished) clears any
  pending TTS and starts recording immediately.

STT backend (set USE_LLAMA_STT below)
  False (default) — local faster-whisper; fastest, no extra server required
  True            — llama.cpp /v1/audio/transcriptions; requires a Whisper
                    model to be served alongside Gemma4

Prerequisites:
    pip install sounddevice numpy pynput python-dotenv
    pip install faster-whisper          # if USE_LLAMA_STT = False
    # Linux: user must be in the 'input' group, or run with sudo, for pynput
    # macOS: grant Accessibility permission to the terminal
"""

from __future__ import annotations

import io
import os
import sys
import wave
import queue
import threading

import numpy as np
import sounddevice as sd
from pynput import keyboard as kb

# ── project path ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from AssistantGlasses.Gemma.chat_cl import Gemma4Agent
from AssistantGlasses.navigation_module.core.nav_controller import NavController
# ─────────────────────────────────────────────────────────────────────────────
# ── Configuration ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE        = 16_000   # Hz — Whisper's native rate; don't change
CHANNELS           = 1
BLOCK_SIZE         = 512      # frames per callback  (~32 ms latency)
WHISPER_MODEL_SIZE = "base"   # "tiny"=fastest  "base"=balanced  "small"=better quality
MIN_RECORD_SECS    = 0.3      # clips shorter than this are silently discarded

# ── STT backend ────────────────────────────────────────────────────────────────
# "whisper_local"   Local faster-whisper model.  Fastest and most reliable.
#                   Default recommendation until Gemma4 native audio stabilises.
#
# "whisper_llama"   llama.cpp /v1/audio/transcriptions endpoint.
#                   Needs a second model served with e.g. --alias whisper.
#
# "gemma4_native"   Send raw WAV directly into Gemma4's audio encoder.
#                   EXPERIMENTAL — requires llama-server ≥ build b8773 and
#                   --mmproj mmproj-BF16.gguf --jinja flags.
#                   llama.cpp tracks active regressions in issues #21868/#23688.
#                   Advantage over Whisper: Gemma4 hears tone and emotional
#                   context, not just the transcribed words.
#                   Auto-falls back to "whisper_llama" on HTTP 500.
#
STT_BACKEND = "whisper_local"   # change me when you're ready to experiment


# ─────────────────────────────────────────────────────────────────────────────
# ── Helpers ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _ndarray_to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert float32 mono numpy array to 16-bit PCM WAV bytes."""
    pcm = (audio.flatten() * 32_767.0).clip(-32_768, 32_767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _flush_stdin() -> None:
    """Discard stdin bytes that piled up while not reading (e.g. SPACE in audio mode)."""
    try:
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        try:
            import msvcrt
            while msvcrt.kbhit():
                msvcrt.getch()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# ── Main class ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class PushToTalk:
    """Push-to-talk console with audio/text mode switching."""

    # ── init ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        agent_role: str = "default",
        agent_host: str = "http://localhost:8090",
        language: str   = "en",
    ) -> None:
        # ── Gemma4 agent ───────────────────────────────────────────────────
        self._dest_q: queue.Queue = queue.Queue()
        self.agent = Gemma4Agent(
            destination = self._dest_q,
            role        = agent_role,
            host        = agent_host,
        )
        self._language = language

        # ── STT backend ────────────────────────────────────────────────────
        if STT_BACKEND == "whisper_local":
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed.\n"
                    "  pip install faster-whisper\n"
                    "Or set STT_BACKEND to 'whisper_llama' or 'gemma4_native'."
                )
            print(f"[mic] Loading faster-whisper ({WHISPER_MODEL_SIZE})…", end=" ", flush=True)
            self._whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            print("ready.")
        elif STT_BACKEND == "whisper_llama":
            self._whisper = None
            print("[mic] Using llama.cpp /v1/audio/transcriptions for STT.")
        elif STT_BACKEND == "gemma4_native":
            self._whisper = None
            print(
                "[mic] Native Gemma4 audio mode (EXPERIMENTAL).\n"
                "      Requires --mmproj mmproj-BF16.gguf --jinja in llama-server.\n"
                "      Will auto-fall back to whisper_llama on errors."
            )
        else:
            raise ValueError(f"Unknown STT_BACKEND: {STT_BACKEND!r}")

        # ── state ──────────────────────────────────────────────────────────
        self.mode          : str  = "audio"
        self._recording    : bool = False
        self._frames       : list = []
        self._rec_lock               = threading.Lock()
        self._agent_busy             = threading.Event()   # set while agent is active
        self._stop_ev                = threading.Event()   # set to quit
        self._space_down   : bool = False

        # ── pynput global keyboard listener ────────────────────────────────
        # suppress=False — let the terminal still see keystrokes so that
        # _stdin_worker can read commands typed in text mode.
        self._listener = kb.Listener(
            on_press   = self._on_key_press,
            on_release = self._on_key_release,
            suppress   = False,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ── Recording ─────────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def _audio_cb(self, indata, frames, t, status) -> None:
        if self._recording:
            self._frames.append(indata.copy())

    def _start_rec(self) -> None:
        with self._rec_lock:
            if self._recording:
                return
            self._frames    = []
            try:
                self._stream = sd.InputStream(
                    samplerate = SAMPLE_RATE,
                    channels   = CHANNELS,
                    dtype      = "float32",
                    blocksize  = BLOCK_SIZE,
                    callback   = self._audio_cb,
                )
                self._stream.start()
                self._recording = True
            except Exception as exc:
                print(f"\n[mic] ✗ Cannot open microphone: {exc}")
                return

        # Interrupt any ongoing TTS the moment the user starts speaking
        self.agent.interrupt_tts()
        print("\r🔴  Recording…  (release SPACE to send)     ", end="", flush=True)

    def _stop_rec(self) -> None:
        with self._rec_lock:
            if not self._recording:
                return
            self._recording = False
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            frames = list(self._frames)

        n_samples = sum(f.shape[0] for f in frames)
        duration  = n_samples / SAMPLE_RATE

        if duration < MIN_RECORD_SECS or not frames:
            print("\r[mic] Too short — ignored.                          ")
            self._prompt()
            return

        audio = np.concatenate(frames).flatten()
        print(f"\r🔈  Transcribing… ({duration:.1f} s)                  ", flush=True)

        # Mark busy *before* spawning the thread to prevent a second
        # recording starting while transcription is in progress.
        self._agent_busy.set()
        threading.Thread(target=self._transcribe, args=(audio,), daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # ── Transcription ─────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def _transcribe(self, audio: np.ndarray) -> None:
        text = ""
        try:
            if STT_BACKEND == "whisper_local":
                segs, _ = self._whisper.transcribe(
                    audio,
                    beam_size  = 1,
                    vad_filter = True,
                    language   = None,   # auto-detect
                )
                text = " ".join(s.text for s in segs).strip()

            elif STT_BACKEND == "whisper_llama":
                wav_bytes = _ndarray_to_wav(audio)
                text = self.agent.transcribe_audio(wav_bytes)

            elif STT_BACKEND == "gemma4_native":
                # Send raw audio straight into Gemma4's encoder.
                # chat_stream() handles auto-fallback to whisper_llama on 500.
                wav_bytes = _ndarray_to_wav(audio)
                if text:
                    print(f"You (voice): {text}")
                self._dispatch_audio(wav_bytes)   # special path — skips text display
                return

        except Exception as exc:
            print(f"[mic] Transcription error: {exc}")

        if text:
            print(f"You (voice): {text}")
            self._dispatch(text)          # _dispatch manages _agent_busy
        else:
            print("[mic] (silence — nothing sent)")
            self._agent_busy.clear()
            self._prompt()

    # ══════════════════════════════════════════════════════════════════════════
    # ── Agent dispatch ────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def _dispatch(self, text: str) -> None:
        """Send text to the agent.  Always runs in a non-main thread."""
        self._agent_busy.set()           # idempotent; also covers text-mode path
        try:
            self.agent.chat_stream(text)
        except Exception as exc:
            print(f"[mic] Agent error: {exc}")
        finally:
            self._agent_busy.clear()
            if not self._stop_ev.is_set():
                self._prompt()

    def _dispatch_audio(self, wav_bytes: bytes) -> None:
        """Send raw WAV bytes directly to Gemma4's native audio encoder."""
        self._agent_busy.set()
        try:
            self.agent.chat_stream(audio_data=wav_bytes, native_audio=True)
        except Exception as exc:
            print(f"[mic] Agent error: {exc}")
        finally:
            self._agent_busy.clear()
            if not self._stop_ev.is_set():
                self._prompt()

    # ══════════════════════════════════════════════════════════════════════════
    # ── pynput callbacks ──────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def _on_key_press(self, key) -> None:
        """Called in pynput's listener thread — must return quickly."""
        if (
            key == kb.Key.space
            and self.mode == "audio"
            and not self._space_down
            and not self._agent_busy.is_set()
        ):
            self._space_down = True
            self._start_rec()

    def _on_key_release(self, key) -> None:
        if key == kb.Key.space and self._space_down:
            self._space_down = False
            # Run _stop_rec in its own thread so the pynput callback returns fast
            threading.Thread(target=self._stop_rec, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    # ── stdin reader (always-on background thread) ────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def _stdin_worker(self) -> None:
        """
        Reads lines from stdin in a loop.
        •  /commands are handled in either mode.
        •  Plain text is dispatched to the agent only in text mode.
        •  In audio mode plain text is silently ignored (space characters that
           accumulate in the terminal buffer while SPACE is held are discarded
           by _flush_stdin() when switching to text mode).
        """
        while not self._stop_ev.is_set():
            try:
                line = sys.stdin.readline()
            except (EOFError, KeyboardInterrupt):
                self._stop_ev.set()
                return

            if not line:               # EOF
                self._stop_ev.set()
                return

            text = line.rstrip("\n").strip()

            if not text:
                if self.mode == "text":
                    print("You: ", end="", flush=True)
                continue

            # ── commands ──────────────────────────────────────────────
            if text.startswith("/"):
                cmd = text[1:].lower()
                if cmd in ("text", "t"):
                    self._switch_mode("text")
                elif cmd in ("audio", "a"):
                    self._switch_mode("audio")
                elif cmd in ("quit", "q", "exit"):
                    self._stop_ev.set()
                    return
                else:
                    print(f"[mic] Unknown command '{text}' — try /audio, /text, /quit.")
                    if self.mode == "text":
                        print("You: ", end="", flush=True)
                continue

            # ── text input ────────────────────────────────────────────
            if self.mode == "text":
                if self._agent_busy.is_set():
                    print("[mic] Still processing previous message, please wait…")
                    print("You: ", end="", flush=True)
                else:
                    threading.Thread(
                        target=self._dispatch, args=(text,), daemon=True
                    ).start()
            # In audio mode, non-command text is ignored

    # ══════════════════════════════════════════════════════════════════════════
    # ── Utilities ─────────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def _switch_mode(self, new_mode: str) -> None:
        if self.mode == new_mode:
            return
        self.mode = new_mode
        if new_mode == "text":
            _flush_stdin()        # discard accumulated SPACE presses from audio mode
        print(f"\n[→ {new_mode.upper()} mode]")
        self._prompt()

    def _prompt(self) -> None:
        if self.mode == "audio":
            print(
                "\n[AUDIO]  Hold SPACE to speak"
                "  │  /text = text mode"
                "  │  /quit = exit"
            )
        else:
            print(
                "\n[TEXT]   Type + Enter to send"
                "  │  /audio = audio mode"
                "  │  /quit = exit"
            )
            print("You: ", end="", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ── Entry point ───────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> None:
        print("=" * 62)
        print("  AssistantGlasses — Push-to-Talk Console")
        print("=" * 62)

        self._listener.start()

        stdin_thread = threading.Thread(
            target=self._stdin_worker, daemon=True, name="stdin-reader"
        )
        stdin_thread.start()

        self._prompt()

        try:
            # Main thread blocks here; all work happens in daemon threads.
            self._stop_ev.wait()
        except KeyboardInterrupt:
            pass
        finally:
            print("\n[mic] Shutting down…")
            self._stop_ev.set()
            self._listener.stop()
            self.agent.stop()
            print("[mic] Goodbye.")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PushToTalk().run()
