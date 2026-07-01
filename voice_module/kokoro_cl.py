
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
    samples, sample_rate = synthesize(text, lan)
    sd.play(samples, sample_rate)
    sd.wait()


if __name__ == "__main__":
    speak(
        "背景看起来像是一个房间或者图书馆的一部分，"
        "右边有很高的书架，书架上放满了书。"
        "天花板上有很多明亮的灯光。"
    )
