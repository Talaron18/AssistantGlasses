import io
import time
import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import base64

class PushToTalkRecorder:
    """
    Hold SPACE -> record
    Release SPACE -> stop
    Returns base64 WAV ready for llama.cpp audio input.
    """
    def __init__(self, samplerate=16000, channels=1, dtype="int16"):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self.audio_chunks = []

    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_chunks.append(indata.copy())

    def start(self):
        self.audio_chunks.clear()
        self.recording = True

    def stop(self):
        self.recording = False
        if not self.audio_chunks:
            return None

        audio = np.concatenate(self.audio_chunks, axis=0)
        buf = io.BytesIO()
        sf.write(buf, audio, self.samplerate, format="WAV")
        wav_bytes = buf.getvalue()

        return {
            "wav_bytes": wav_bytes,
            "base64": base64.b64encode(wav_bytes).decode(),
        }

    def listen(self):
        print("\n[MIC] 🎤 Ready! Hold SPACE to talk...")
        with sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        ):
            while True:
                keyboard.wait("space")
                print("\r[MIC] 🔴 Recording...", end="", flush=True)
                self.start()
                
                while keyboard.is_pressed("space"):
                    time.sleep(0.02)
                
                result = self.stop()
                print("\r[MIC] 🟢 Finished.   ", flush=True)
                
                if result:
                    yield result