from kokoro_onnx import Kokoro
from misaki import zh, en
import sounddevice as sd
import os
from dotenv import load_dotenv

load_dotenv()

_global_kokoro = None
_global_g2p = None
_global_voice = None

def init_tts(lan="default"):
    global _global_kokoro, _global_g2p, _global_voice
    
    if _global_kokoro is not None:
        return

    print("[TTS] Initializing Kokoro TTS models...")
    if lan in("zh", "default"):
        _global_g2p = zh.ZHG2P(version="1.1")
        _global_voice = "zf_001"
        model_path = os.environ.get("ONNX-ZH")
        voice_path = os.environ.get("BIN-ZH")
        config = os.environ.get("CONFIG-ZH")
        _global_kokoro = Kokoro(model_path=model_path, voices_path=voice_path, vocab_config=config)
       
    elif lan == "en":
        _global_g2p = en.G2P(version="1.1")
        _global_voice = "zf_001"
        model_path = os.environ.get("ONNX-EN")
        voice_path = os.environ.get("BIN-EN")
        _global_kokoro = Kokoro(model_path=model_path, voices_path=voice_path)
    else:
        print("[TTS] Language not supported")
        exit()
    print("[TTS] Initialization complete.")

def speak(text, lan="default"):
    global _global_kokoro, _global_g2p, _global_voice
    
    if _global_kokoro is None:
        init_tts(lan)
        
    try:
        phonemes, _ = _global_g2p(text)
        samples, sr = _global_kokoro.create(
            phonemes,
            voice=_global_voice,
            speed=1.5,
            is_phonemes=True,
        )
        
        sd.play(samples, sr)
        sd.wait() 
    except Exception as e:
        print(f"\n[TTS Error] {e}")

if __name__=="__main__":
    init_tts("zh")
    speak("背景看起来像是一个房间或者图书馆的一部分，右边有很高的书架，书架上放满了书。天花板上有很多明亮的灯光。")