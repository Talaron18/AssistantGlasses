## Auto Speech Recognition Module
### Introduction
This is a simple implementation of OpenAI's speech-to-text model [whisper-base](https://huggingface.co/openai/whisper-base). Due to hardware limitation, we adopted the [openvino_genai pipeline](https://github.com/openvinotoolkit/openvino.genai/tree/master/samples/python/whisper_speech_recognition) and added a "wake-word" detection based on [porcupine](https://github.com/Picovoice/porcupine) together with [cobra](https://github.com/Picovoice/cobra)
### Hierarchy
```mermaid
graph TD
    %% Main Project Root
    Root[speech_module/] --> Stream[stream/]
    Root --> Tests[tests/]
    Root --> Install(install.py)
    Root --> RM(README.md)
    Root --> REQ(requirements.txt)

    %% Stream Directory
    Stream --> Act(activate.py)
    Stream --> Rec(record.py)
    Stream --> Util(utils.py)

    %% Tests Directory
    Tests --> Wake(wake_word.py)
    Tests --> SR(speech_recognition.py)

    %% Styling for visual clarity
    style Root fill:#fdf,stroke:#333,stroke-width:2px
    style Stream fill:#eef,stroke:#333
    style Tests fill:#eef,stroke:#333
```
### Details
Download [language models](https://github.com/Picovoice/porcupine/tree/master/lib/common) from porcupine in advance for "wake word" operation. You may also need to visit the official website for [Picovoice](https://picovoice.ai/) to apply for an access key to use their products.
***
Run [wake_word.py](tests\wake_word.py) to try "keyword wake up".

Run [speech_recognition.py](tests\speech_recognition.py) to try ASR.