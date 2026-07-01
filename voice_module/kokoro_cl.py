"""
kokoro.py — Cached Kokoro TTS with a pipelined synthesize/speak interface.

Key changes vs original:
  - Models and G2P processors are loaded ONCE at first use, then reused.
    The per-call Kokoro() constructor was the dominant latency cost; this
    eliminates it entirely on subsequent calls.
  - synthesize(text, lan) → (samples, rate)  — generates audio WITHOUT playing.
    chat.py uses this to pipeline generation and playback in separate threads
    so chunk N+1 is synthesised while chunk N is playing.
  - speak(text, lan)  — backward-compatible; calls synthesize then plays.
  - English model constructor bug fixed (voice_path kept, vocab_config optional).
"""

from __future__ import annotations

import os
import threading

import sounddevice as sd
from dotenv import load_dotenv
from kokoro_onnx import Kokoro
from misaki import en, zh

load_dotenv()


_lock:   threading.Lock          = threading.Lock()
_kokoro: dict[str, Kokoro]       = {}
_g2p:    dict[str, object]       = {}


def _ensure_zh() -> tuple[Kokoro, zh.ZHG2P]:
    """Load Chinese model + G2P exactly once."""
    with _lock:
        if "zh" not in _kokoro:
            print("[TTS] Loading Chinese model…", end=" ", flush=True)
            _g2p["zh"]    = zh.ZHG2P(version="1.1")
            _kokoro["zh"] = Kokoro(
                model_path   = os.environ["ONNX-ZH"],
                voices_path  = os.environ["BIN-ZH"],
                vocab_config = os.environ["CONFIG-ZH"],
            )
            print("ready.")
    return _kokoro["zh"], _g2p["zh"]


def _ensure_en() -> tuple[Kokoro, en.G2P]:
    """Load English model + G2P exactly once."""
    with _lock:
        if "en" not in _kokoro:
            print("[TTS] Loading English model…", end=" ", flush=True)
            _g2p["en"] = en.G2P(version="1.1")

            kwargs: dict = {
                "model_path": os.environ["ONNX-EN"],
                "voices_path": os.environ["BIN-EN"],
            }

            _kokoro["en"] = Kokoro(**kwargs)
            print("ready.")
    return _kokoro["en"], _g2p["en"]


def synthesize(text: str, lan: str = "default") -> tuple:
    """
    Convert *text* to raw audio samples WITHOUT playing.

    Returns
    -------
    (samples : np.ndarray, sample_rate : int)

    Designed to be called from a synthesis thread so that audio generation
    overlaps with playback of the previous chunk.
    """
    if not text or not text.strip():
        raise ValueError("synthesize() received empty text.")

    if lan in ("zh", "default"):
        kokoro, g2p = _ensure_zh()
        phonemes, _ = g2p(text)
        return kokoro.create(phonemes, voice="zf_001", speed=1.0, is_phonemes=True)

    if lan == "en":
        kokoro, g2p = _ensure_en()
        phonemes, _ = g2p(text)
        return kokoro.create(phonemes, "af_heart", is_phonemes=True)

    raise ValueError(f"[TTS] Language '{lan}' is not supported.")


def speak(text: str, lan: str = "default") -> None:
    """Synthesise *text* and play it immediately (blocking).  Backward-compatible."""
    samples, sample_rate = synthesize(text, lan)
    sd.play(samples, sample_rate)
    sd.wait()


if __name__ == "__main__":
    speak(
        "背景看起来像是一个房间或者图书馆的一部分，"
        "右边有很高的书架，书架上放满了书。"
        "天花板上有很多明亮的灯光。"
    )
