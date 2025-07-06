import pygame
import time

pygame.init()
print("播放音乐1")
pygame.mixer.music.load(r"meter_reading_res/warning.wav")

pygame.mixer.music.play()
time.sleep(10)
pygame.mixer.music.stop()