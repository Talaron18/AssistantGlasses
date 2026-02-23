import edge_tts
import pygame
import os

async def play(text,voice):
    audio_file="temp_response.mp3"
    communicate=edge_tts.Communicate(text,voice)
    await communicate.save(audio_file)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock.tick(10)
    pygame.mixer.music.unload()
    if os.path.exists(audio_file):
        os.remove(audio_file)