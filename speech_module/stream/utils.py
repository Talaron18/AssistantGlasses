import struct
import numpy as np
import openvino_genai
import os
import noisereduce as nr
from datetime import timedelta
# keyword check 
def wake(handle, pcm_bytes):
    pcm_ints = struct.unpack_from("h" * handle.frame_length, pcm_bytes)
    return handle.process(pcm_ints) == 0

def recognition(pipe, audio_buffer,rate,denoise=False):
    print(f"Denoise settings:{denoise}")
    if audio_buffer is None or len(audio_buffer)==0:
        print("Invalid input detected...")
        return ""
    # set "denoise=True" to activate denoise function
    if denoise:
        audio_ready=denoise(audio_buffer,rate)
    else:
        audio_ready=audio_buffer
    audio_float32 = audio_ready.astype(np.float32) / 32768.0
    try:
        # do not mix multiple languages
        # follow the format: "<|**|>" to set languages
        # eg. "<|en|>""<|de|>""<|fr|>""<|ja|>""<|zh|>" ISO-639-1
        result = pipe.generate(audio_float32.tolist(),language="<|zh|>")
    except Exception as e:
        print(f"Error occurred:{e}")
        import traceback
        traceback.print_exc()
    return result.texts[0]

def voice_to_text(frames,rate):
    whisper=os.environ.get("WHISPER_DIR")
    print("Model path found..." if whisper else "Model not found...")
    pipe=openvino_genai.WhisperPipeline(whisper,"GPU")
    if pipe is not None:
        # turn on denoise function here if needed
        print("Whisper initiated...")
        text=recognition(pipe,frames,rate)
    else:
        print("Pipe Invalid...")
    return text

def denoise(audio_int16,rate):
    processed_audio=nr.reduce_noise(y=audio_int16,sr=rate,prop_decrease=0.6)
    return processed_audio

"""
    ATTENTION! gpiod is only available on Linux
"""

def setup_button(chip_path, pin): # find chippath with "sudo gpioinfo"
    import gpiod
    settings = gpiod.LineSettings(
        edge_detection=gpiod.line.Edge.FALLING,
        bias=gpiod.line.Bias.PULL_UP,
        debounce_period=timedelta(milliseconds=50)
    )
    return gpiod.request_lines(chip_path, consumer="button", config={pin: settings})

def manual_close(handle, pcm_bytes, mode="voice", button_request=None):
    if mode == "voice":
        pcm_ints = struct.unpack_from("h" * handle.frame_length, pcm_bytes)
        return handle.process(pcm_ints) > 0
    elif mode == "button" and button_request:
        if button_request.wait_edge_events(timedelta(seconds=0)):
            # Clear the event from the buffer
            button_request.read_edge_events() 
            return True
    return False

def keyword_check(handle,pcm_bytes):
    pcm_ints = struct.unpack_from("h" * handle.frame_length, pcm_bytes)
    if pcm_ints == 1:
        keyword = "to_agent"
    elif pcm_ints == 2:
        keyword = "stop_listening"
    elif pcm_ints == 3:
        keyword = "navigation"
    return keyword