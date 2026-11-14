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

pygame.init()

async def main():
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    pygame.display.set_caption("Boba Comeback")
    clock = pygame.time.Clock()
    CUSTOMERS = {
        "hamster":{"default":pygame.image.load("assets/characters/hamster_default.jpg"), "sad":pygame.image.load("assets/characters/hamster_disgusted.jpg"), "happy":pygame.image.load("assets/characters/hamster_happy.jpg")},
    }

    class Customer:
        pass

    print("CV_ON:", CV_ON)

    camera = None
    use_camera = False      # True = OpenCV mode, False = clicking mode

    font = pygame.font.Font(None, 40)
    browser_button = pygame.Rect(240, 300, 360, 120)
    download_button = pygame.Rect(680, 300, 360, 120)
    mode_message = ""

    def start_camera():
        nonlocal camera, use_camera, mode_message
        if not CV_ON:
            mode_message = "Webcam not available here. Using clicking mode."
            return
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("camera opened:", camera.isOpened())
        if camera.isOpened():
            use_camera = True
        else:
            mode_message = "Couldn't open webcam. Using clicking mode."

    class Topping(pygame.sprite.Sprite):
        def __init__(self, image_path, x, y, value):
            super().__init__()
            self.image = pygame.image.load(image_path).convert_alpha()
            self.rect = self.image.get_rect(topleft=(x, y))
            self.value = value

            def update(self):
                pass
            def handle_click(self):
                nonlocal money
                money += self.value
                self.kill()

    #MAIN#
    money = 0

    status = "mode_select"

    toppings = pygame.sprite.Group()
    #toppings.add(Topping("assets/toppings/strawberry.png", 100, 200, 5))
    #toppings.add(Topping("assets/toppings/pearls.png", 300, 200, 10))
    #toppings.add(Topping("assets/toppings/matcha.png", 500, 200, 15))

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
            elif event.type == MOUSEBUTTONDOWN:
                if status == "mode_select":
                    if browser_button.collidepoint(event.pos):
                        status = "title"
                    elif download_button.collidepoint(event.pos):
                        start_camera()
                        status = "title"
                elif status == "game":
                    if not use_camera:
                        clicked = False
                        for topping in reversed(toppings.sprites()):
                            if topping.rect.collidepoint(event.pos) and not clicked:
                                topping.handle_click()
                                clicked = True

        screen.fill((0, 0, 0))

        if use_camera:
            ok, frame = camera.read()
            if ok:
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_image = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
                screen.blit(camera_image, (0, 0))

        if status == "mode_select":
            pygame.draw.rect(screen, (255, 255, 255), browser_button, 3)
            pygame.draw.rect(screen, (255, 255, 255), download_button, 3)
            browser_text = font.render("Browser version", True, (255, 255, 255))
            download_text = font.render("Downloaded version", True, (255, 255, 255))
            screen.blit(browser_text, browser_text.get_rect(center=browser_button.center))
            screen.blit(download_text, download_text.get_rect(center=download_button.center))
        elif status == "title":
            if mode_message != "":
                message_text = font.render(mode_message, True, (255, 255, 255))
                screen.blit(message_text, message_text.get_rect(center=(640, 650)))
            #screen.blit(title_image,(0,0))
            pass
        elif status == "tutorial":
            pass
        elif status == "game":
            if use_camera:
                # OPENCV MODE (Sophia): webcam controls for toppings here
                pass
            else:
                toppings.update()
                toppings.draw(screen)

            money_text = font.render(f"Money: ${money}", True, (255, 255, 255))
            screen.blit(money_text, (20, 20))

            #WHEN ADDING MORE TOPPINGS:
            #toppings.add(Topping("assets/toppings/example.png", 700, 300, 20))

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    if camera is not None:
        camera.release()
    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())