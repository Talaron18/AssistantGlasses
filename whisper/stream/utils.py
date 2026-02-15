import struct
import numpy as np
import openvino_genai
import os
import noisereduce as nr

# keyword check 
def wake(handle, pcm_bytes):
    pcm_ints = struct.unpack_from("h" * handle.frame_length, pcm_bytes)
    return handle.process(pcm_ints) >= 0

def recognition(pipe, audio_buffer,rate,denoise=False):
    if not audio_buffer:
        return ""
    # set "denoise=True" to activate denoise function
    if denoise:
        clean_audio=denoise(audio_buffer,rate)
    audio_float32 = np.concatenate(clean_audio).astype(np.float32) / 32768.0
    print("AI is thinking...")
    result = pipe.generate(audio_float32.tolist())
    return result.texts[0]

def voice_to_text(frames,rate):
    whisper=os.environ.get("WHISPER_DIR")
    pipe=openvino_genai.WisperPipeline(whisper,"GPU")
    # turn on denoise function here if needed
    text=recognition(pipe,frames,rate)
    return text

def denoise(audio_int16,rate):
    processed_audio=nr.reduce_noise(y=audio_int16,sr=rate,prop_decrease=0.6)
    return processed_audio