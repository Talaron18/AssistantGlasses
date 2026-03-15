import pyttsx3

class TTS():
    def __init__(self):
        self.engine=pyttsx3.init()
        self.engine.setProperty("rate",200)

    def speak(self,text:str):
        self.engine.say(text)
        self.engine.runAndWait()