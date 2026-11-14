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
    CUSTOMERS = {
        "hamster":{"default":pygame.image.load("assets/characters/hamster_default.png"), "sad":pygame.image.load("assets/character/hamster_disgusted.png"), "happy":pygame.image.load("assets/hamster_happy.png")},
    }

    class Customer:
        pass

    print("CV_ON:", CV_ON)

    camera = None
    if CV_ON:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("camera opened:", camera.isOpened())

    class Topping:
        def __init__(self, image, center_pos, size=120):
            self.image = pygame.image.load(image)
            self.size = size

            self.rect = self.image.get_rect(center=center_pos)

        def draw(self, screen):
            screen.blit(self.image, self.rect)
            pygame.draw.rect(screen, (80, 80, 80), self.rect, 3)

        def clicked(self, pos):
            return self.rect.collidepoint(pos)


    
    #MAIN#
    money = 0

    status = "title"
    toppings = [
        #add Topping() here
    ]
    tutorial_text = [

    ]

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_RETURN and status == "title":
                    status = "tutorial"
                elif event.key == K_RETURN and status == "tutorial":
                    status = "game"

        screen.fill((0, 0, 0))

        if camera is not None:
            ok, frame = camera.read()
            if ok:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_image = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
                screen.blit(camera_image, (0, 0))
        if status == "title":
            #screen.blit(title_image,(0,0))
            pass
        elif status == "tutorial":
            pass
        elif status == "game":
            for topping in toppings:
                topping.update()
                topping.draw(screen)
            pass

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    if camera is not None:
        camera.release()
    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())