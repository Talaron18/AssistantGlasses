import edge_tts
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import AssistantGlasses.voice_module.config as config
async def stream_audio(text, process):
    communicate = edge_tts.Communicate(text, config.VOICES[0])
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
                try:
                    process.stdin.write(chunk["data"])
                    process.stdin.flush()
                except BrokenPipeError:
                    break