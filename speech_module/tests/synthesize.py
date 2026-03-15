import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from AssistantGlasses.speech_module.stream.record import stream

if __name__=="__main__":
    stream()