import edge_tts

async def stream_audio(text, process):
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
                try:
                    process.stdin.write(chunk["data"])
                    process.stdin.flush()
                except BrokenPipeError:
                    break