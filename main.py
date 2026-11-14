import pygame
from pygame import *
import asyncio
import random

try:
    import cv2
    CV_ON = True
except ImportError as error:
    print("OpenCV error:", error)
    CV_ON = False

async def main():
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    pygame.display.set_caption("Boba Comeback")
    clock = pygame.time.Clock()

    print("CV_ON:", CV_ON)

    camera = None
    if CV_ON:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("camera opened:", camera.isOpened())

    #MAIN#

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

        screen.fill((0, 0, 0))

        if camera is not None:
            ok, frame = camera.read()
            if ok:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_image = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
                screen.blit(camera_image, (0, 0))

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    if camera is not None:
        camera.release()
    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())