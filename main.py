import pygame
from pygame import *
import asyncio
import random

try:
    import cv2
    from mp_palmdet import MPPalmDet
    from mp_handpose import MPHandPose
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
        "hamster":{"default":pygame.image.load("assets/characters/hamster_default.png"), "angry":pygame.image.load("assets/characters/hamster_angry.png"), "happy":pygame.image.load("assets/characters/hamster_happy.png")},
        "chiikawa":{"default":pygame.image.load("assets/characters/chiikawa_default.png"), "angry":pygame.image.load("assets/characters/chiikawa_angry.png"), "happy":pygame.image.load("assets/characters/chiikawa_happy.png")},
        "peppapig":{"default":pygame.image.load("assets/characters/peppapig_default.png"), "angry":pygame.image.load("assets/characters/peppapig_angry.png"), "happy":pygame.image.load("assets/characters/peppapig_happy.png")},
        "redpanda":{"default":pygame.image.load("assets/characters/redpanda_default.png"), "angry":pygame.image.load("assets/characters/redpanda_angry.png"), "happy":pygame.image.load("assets/characters/redpanda_happy.png")},
    }
    TOPPINGS = {
        #"pearls": {"image": pygame.image.load("assets/toppings/pearls.png"), "value": 10},
    }

    print("CV_ON:", CV_ON)

    camera = None
    use_camera = False      # True = OpenCV mode, False = clicking mode

    # hand detection (OpenCV mode only)
    palm_detector = None
    hand_detector = None
    finger_count = 0        # fingers counted this frame (both hands)
    held_count = 0          # the number being held right now
    hold_frames = 0         # how long it's been held
    HOLD_TIME = 45          # 45 frames = about 1 second
    waiting_for_reset = False

    # finger number -> topping. Names must match the keys in TOPPINGS
    HAND_TOPPINGS = ["pearls", "grass jelly", "coconut jelly", "strawberry popping boba",
                     "mango popping boba", "mango chunks", "strawberry chunks", "red bean"]

    font = pygame.font.Font(None, 40)
    browser_button = pygame.Rect(240, 300, 360, 120) # kinda chopped rn so we can make it look better when we have artists UI done/titlepage
    download_button = pygame.Rect(680, 300, 360, 120)
    mode_message = ""

    def start_camera():
        nonlocal camera, use_camera, mode_message, palm_detector, hand_detector
        if not CV_ON:
            mode_message = "Webcam not available here. Using clicking mode."
            return
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("camera opened:", camera.isOpened())
        if camera.isOpened():
            use_camera = True
            palm_detector = MPPalmDet("assets/models/palm_detection_mediapipe_2023feb.onnx", scoreThreshold=0.6)
            hand_detector = MPHandPose("assets/models/handpose_estimation_mediapipe_2023feb.onnx")
        else:
            mode_message = "Couldn't open webcam. Using clicking mode."

    def distance(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def count_fingers_on_hand(points):
        # a finger is up if its tip is farther from the wrist than its middle joint
        wrist = points[0]
        count = 0
        for tip, joint in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if distance(points[tip], wrist) > distance(points[joint], wrist):
                count += 1
        # thumb is out if its tip is farther from the middle finger's base than its joint
        if distance(points[4], points[9]) > distance(points[3], points[9]):
            count += 1
        return count

    def count_all_fingers(frame):
        total = 0
        palms = palm_detector.infer(frame)
        for palm in palms[:2]:                       # only the first 2 hands
            hand = hand_detector.infer(frame, palm)
            if hand is not None:
                points = hand[4:67].reshape(21, 3)    # 21 points, each (x, y, z)
                total += count_fingers_on_hand(points)
        return total

    def add_topping_by_hand(name):
        if name not in selected_toppings:
            selected_toppings.append(name)

    def picking_toppings():
        # hand motions only work while a customer is waiting and the toppings aren't done
        if customer is None or customer_reacting:
            return False
        if len(selected_toppings) >= len(current_order) or len(selected_toppings) >= 3:
            return False
        return True

    # CUSTOMER DIALOGUE (Sophia)
    def fill_in_dialogue(line):
        toppings_text = " and ".join(current_order)
        return line.format(base=current_base, toppings=toppings_text, topping=toppings_text)

    def get_mood():
        # happy = right order, meh = some toppings right, angry = none right
        if set(selected_toppings) == set(current_order):
            return "happy"
        for name in selected_toppings:
            if name in current_order:
                return "meh"
        return "angry"

    def wrap_text(text, font, max_width):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = (current_line + " " + word).strip()
            if font.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    def generate_order():
        number = random.randint(1, len(available_toppings))
        return random.sample(available_toppings, number)

    def new_customer():
        nonlocal customer, current_order

        name = random.choice(list(CUSTOMERS.keys()))
        current_order = generate_order()
        customer = Customer(name, current_order)

        selected_toppings.clear()
        toppings.empty()

        positions = [
            (300, 450),
            (500, 450),
            (700, 450)
        ]

        for index, topping_name in enumerate(available_toppings):
            x, y = positions[index]
            toppings.add(Topping(
                TOPPINGS[topping_name]["image"],
                x,
                y,
                topping_name,
                TOPPINGS[topping_name]["value"]
            ))

    class Customer:
            def __init__(self, name, order):
                self.name = name
                self.order = order
                self.image = CUSTOMERS[name]["default"]
                self.rect = self.image.get_rect(center=(640, 200))
    
            def check_order(self, selected_toppings):
                if set(selected_toppings) == set(self.order):
                    self.image = CUSTOMERS[self.name]["happy"]
                    return True
                else:
                    self.image = CUSTOMERS[self.name]["angry"]
                    return False
    
            def draw(self, screen):
                screen.blit(self.image, self.rect)

    class Topping(pygame.sprite.Sprite):
        def __init__(self, image_path, x, y, name, value):
            super().__init__()
            self.image = pygame.image.load(image_path).convert_alpha()
            self.rect = self.image.get_rect(topleft=(x, y))
            self.value = value
            self.name = name

            def update(self):
                pass
            def handle_click(self):
                nonlocal money
                money += self.value
                self.kill()

        def draw(self, screen):
            screen.blit(self.image, self.rect)
            pygame.draw.rect(screen, (80, 80, 80), self.rect, 3)

        def clicked(self, pos):
            return self.rect.collidepoint(pos)
    
    #MAIN#
    money = 0
    status = "mode_select"

    toppings = pygame.sprite.Group()

    available_toppings = ['''add three here''']
    selected_toppings = []
    current_order = []
    customer = None

    customer_timer = 0
    next_customer_time = random.randint(3000, 7000)

    customer_reacting = False
    reaction_timer = 0

    cutscene = [pygame.image.load("assets/ui/cutscene_frame1.png"), pygame.image.load("assets/ui/cutscene_frame2.png")]
    cutscene_index = 0

    ending_pending = False

    # MAIN SCREEN BACKGROUND (Sophia)
    main_screen = pygame.transform.scale(pygame.image.load("assets/ui/background.png").convert(), (SCREEN_WIDTH, SCREEN_HEIGHT))
    camera_image = None


    # CUSTOMER DIALOGUE (Sophia)
    # names must match the keys in CUSTOMERS
    BASES = ["milk tea", "taro milk tea", "matcha"]
    CUSTOMER_DIALOGUE = {
        "snoopy": {
            "order": ["A {base} with {toppings}... the perfect reward after a long day of napping."],
            "happy": ["*happy dance* ...This {base} is perfect. Best nap snack ever."],
            "meh": ["*tilts head* ...Not bad. Not great. Like a nap that was too short."],
            "angry": ["*flops over* ...This is NOT what I ordered. Ugh."],
        },
        "hamster": {
            "order": ["HI CHAT CAN I GET A {base} WITH SOME {topping} PLSPSLPSSPLS!!!!"],
            "happy": ["I LOVE THIS DRINK AND I LOVE YOUUUUUUU BYEEE"],
            "meh": ["Hmm, I think you made it wrong... that's okay ig."],
            "angry": ["Bro. Are u fr. I'm allergic to {base}"],
        },
        "redpanda": {
            "order": ["Grr uwu"],
            "happy": ["Grr~ uwu! It's perfect, thank youuu!"],
            "meh": ["Grr... it's okay I guess... uwu"],
            "angry": ["GRRR! This is NOT what I asked for! >:("],
        },
        "chiikawa": {
            "order": ["Hmph! I-it's not like I wanted a {base} with {toppings} or anything! And I definitely didn't come here just to see you... b-baka!"],
            "happy": ["Wow you made it right... but it's not like it was for me, s-so I don't care or anything!!!! H-hmph~"],
            "meh": ["Th-this is kind of close... b-but don't think I'm impressed or anything! Hmph."],
            "angry": ["b-baka! What the hell's wrong with you! Can't even make a simple drink..!"],
        },
        "peppapig": {
            "order": ["Two Two's my Poo Poo's, I finna take a {base} w some {topping}, don't be a Gerbert fam."],
            "happy": ["Yooo this is fire! Bless up bossman!"],
            "meh": ["Nah who made this. I'm sooper cheesed fam."],
            "angry": ["cro... wrong order fam."],
        },
    }
    dialogue_font = pygame.font.Font(None, 30)
    dialogue_box = pygame.Rect(340, 320, 600, 110)
    dialogue_customer = None     # the customer the current dialogue belongs to
    dialogue_text = ""
    current_base = ""
    reaction_given = False

    serve_button = pygame.Rect(1000, 600, 200, 70)

    running = True

    while running:
        dt = clock.get_time()

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
                elif event.key == K_SPACE and status == "tutorial":
                    if event.key == K_SPACE and status == "tutorial":
                        if cutscene_index < len(cutscene) - 1:
                            cutscene_index += 1
            elif event.type == MOUSEBUTTONDOWN:
                if status == "mode_select":
                    if browser_button.collidepoint(event.pos):
                        status = "title"
                    elif download_button.collidepoint(event.pos):
                        start_camera()
                        status = "title"
                elif status == "game":
                    if not use_camera and customer is not None and not customer_reacting:
                            clicked = False

                            for topping in reversed(toppings.sprites()):
                                if topping.rect.collidepoint(event.pos) and not clicked:
                                    topping.handle_click()
                                    clicked = True

                            if serve_button.collidepoint(event.pos):
                                accepted = customer.check_order(selected_toppings)

                                customer_reacting = True
                                reaction_timer = 0

                                if accepted:
                                    money += sum(
                                        TOPPINGS[name]["value"]
                                        for name in current_order
                                    )

                    # OPENCV MODE (Sophia): serve is still done with the mouse
                    if use_camera and customer is not None and not customer_reacting:
                        if serve_button.collidepoint(event.pos):
                            accepted = customer.check_order(selected_toppings)

                            customer_reacting = True
                            reaction_timer = 0

                            if accepted:
                                money += sum(
                                    TOPPINGS[name]["value"]
                                    for name in current_order
                                )

        screen.fill((0, 0, 0))


        screen.fill((0, 0, 0))

        if use_camera:
            ok, frame = camera.read()
            if ok:
                frame = cv2.flip(frame, 1)
                if status == "game" and picking_toppings():
                    finger_count = count_all_fingers(frame)
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
            screen.blit(cutscene[cutscene_index], (0, 0))
        elif status == "game":
            # MAIN SCREEN BACKGROUND (Sophia)
            screen.blit(main_screen, (0, 0))
            if use_camera and camera_image is not None:
                # small webcam preview so players can still see their hands
                screen.blit(pygame.transform.scale(camera_image, (320, 240)), (SCREEN_WIDTH - 330, 10))

            if use_camera:
                # OPENCV MODE (Sophia): same as clicking mode, but toppings are picked with fingers
                if customer is None and not customer_reacting:
                    customer_timer += dt

                    if customer_timer >= next_customer_time:
                        new_customer()
                        customer_timer = 0

                # CUSTOMER REACTION
                if customer_reacting:
                    reaction_timer += dt

                    if reaction_timer >= 2000:
                        customer = None
                        customer_reacting = False
                        reaction_timer = 0

                        if ending_pending:
                            status = "ending"
                        else:
                            customer_timer = 0
                            next_customer_time = random.randint(3000, 7000)

                # DRAW CUSTOMER
                if customer is not None:
                    customer.draw(screen)

                # PICK TOPPINGS WITH FINGERS
                if picking_toppings():
                    if finger_count == 0:
                        waiting_for_reset = False
                        hold_frames = 0
                    elif waiting_for_reset:
                        pass
                    elif finger_count == held_count:
                        hold_frames += 1
                        if hold_frames >= HOLD_TIME and finger_count <= len(HAND_TOPPINGS):
                            add_topping_by_hand(HAND_TOPPINGS[finger_count - 1])
                            waiting_for_reset = True
                            hold_frames = 0
                    else:
                        held_count = finger_count
                        hold_frames = 0

                    fingers_text = font.render("Fingers: " + str(finger_count), True, (255, 255, 255))
                    screen.blit(fingers_text, (20, 520))
                    pygame.draw.rect(screen, (255, 255, 255), (20, 560, 300, 25), 2)
                    pygame.draw.rect(screen, (255, 255, 255), (20, 560, int(300 * hold_frames / HOLD_TIME), 25))

                # TOPPINGS DONE
                elif customer is not None and not customer_reacting:
                    done_text = font.render("Toppings done! Click SERVE", True, (255, 255, 255))
                    screen.blit(done_text, (20, 520))

                # SERVE BUTTON
                pygame.draw.rect(screen, (255, 255, 255), serve_button, 3)
                serve_text = font.render("SERVE", True, (255, 255, 255))
                screen.blit(serve_text, serve_text.get_rect(center=serve_button.center))

                # DISPLAY SELECTED TOPPINGS
                selected_text = font.render(
                    "Selected: " + ", ".join(selected_toppings),
                    True,
                    (255, 255, 255)
                )
                screen.blit(selected_text, (20, 70))

                # DISPLAY ORDER (TEMPORARY FOR TESTING)
                if customer is not None:
                    order_text = font.render(
                        "Order: " + ", ".join(current_order),
                        True,
                        (255, 255, 255)
                    )
                    screen.blit(order_text, (20, 120))
            else:
                if customer is None and not customer_reacting:
                    customer_timer += dt

                    if customer_timer >= next_customer_time:
                        new_customer()
                        customer_timer = 0

                # CUSTOMER REACTION
                if customer_reacting:
                    reaction_timer += dt
                
                    if reaction_timer >= 2000:
                        customer = None
                        customer_reacting = False
                        reaction_timer = 0
                
                        if ending_pending:
                            status = "ending"
                        else:
                            customer_timer = 0
                            next_customer_time = random.randint(3000, 7000)

                # DRAW CUSTOMER
                if customer is not None:
                    customer.draw(screen)

                # DRAW TOPPINGS
                toppings.update()
                toppings.draw(screen)

                # SERVE BUTTON
                pygame.draw.rect(screen, (148, 90, 45), serve_button, 3)
                serve_text = font.render("SERVE", True, (148, 90, 45))
                screen.blit(serve_text, serve_text.get_rect(center=serve_button.center))

                # DISPLAY SELECTED TOPPINGS
                selected_text = font.render(
                    "Selected: " + ", ".join(selected_toppings),
                    True,
                    (255, 255, 255)
                )
                screen.blit(selected_text, (20, 70))

                # DISPLAY ORDER (TEMPORARY FOR TESTING)
                if customer is not None:
                    order_text = font.render(
                        "Order: " + ", ".join(current_order),
                        True,
                        (255, 255, 255)
                    )
                    screen.blit(order_text, (20, 120))

            # CUSTOMER DIALOGUE (Sophia)
            # new customer walks up: pick a base and an order line
            if customer is not None and customer is not dialogue_customer:
                dialogue_customer = customer
                current_base = random.choice(BASES)
                reaction_given = False
                dialogue_text = ""
                if customer.name in CUSTOMER_DIALOGUE:
                    dialogue_text = fill_in_dialogue(random.choice(CUSTOMER_DIALOGUE[customer.name]["order"]))

            # drink was served: switch to a reaction line
            if customer is not None and customer_reacting and not reaction_given:
                reaction_given = True
                if customer.name in CUSTOMER_DIALOGUE:
                    mood = get_mood()
                    dialogue_text = fill_in_dialogue(random.choice(CUSTOMER_DIALOGUE[customer.name][mood]))

            # customer left
            if customer is None:
                dialogue_text = ""

            # DRAW DIALOGUE
            if dialogue_text != "":
                pygame.draw.rect(screen, (148, 90, 45), dialogue_box)
                pygame.draw.rect(screen, (80, 80, 80), dialogue_box, 3)
                y = dialogue_box.top + 15
                for line in wrap_text(dialogue_text, dialogue_font, dialogue_box.width - 30):
                    line_surface = dialogue_font.render(line, True, (0, 0, 0))
                    screen.blit(line_surface, (dialogue_box.left + 15, y))
                    y += 28

            money_text = font.render(f"Money: ${money}", True, (148, 90, 45))
            screen.blit(money_text, (20, 20))

            # WHEN ADDING MORE TOPPINGS:
            # toppings.add(Topping("assets/toppings/example.png", 700, 300, "example", 20))
        elif status == "ending":
            screen.fill((255, 230, 200))

            ending_text = font.render(
                "You reached $100! Congratulations!", True, (148, 90, 45))

            screen.blit(ending_text, ending_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)))

        if money >= 100 and status == "game":
            ending_pending = True

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    if camera is not None:
        camera.release()
    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())