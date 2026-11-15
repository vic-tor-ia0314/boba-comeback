import pygame
from pygame import *
import asyncio
import random
import math

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
        "chiikawa":{"default":pygame.image.load("assets/characters/chiikawa_default.png"), "angry":pygame.image.load("assets/characters/chiikawa_angry.png"), "happy":pygame.image.load("assets/characters/chiikawa_happy.png"), "meh":pygame.image.load("assets/characters/chiikawa_meh.png")},
        "peppapig":{"default":pygame.image.load("assets/characters/peppapig_default.png"), "angry":pygame.image.load("assets/characters/peppapig_angry.png"), "happy":pygame.image.load("assets/characters/peppapig_happy.png"), "meh":pygame.image.load("assets/characters/peppapig_meh.png")},
        "redpanda":{"default":pygame.image.load("assets/characters/redpanda_default.png"), "angry":pygame.image.load("assets/characters/redpanda_angry.png"), "happy":pygame.image.load("assets/characters/redpanda_happy.png"), "meh":pygame.image.load("assets/characters/redpanda_meh.png")},
        "snoopy":{"default":pygame.image.load("assets/characters/snoopy_default.png"), "angry":pygame.image.load("assets/characters/snoopy_angry.png"), "happy":pygame.image.load("assets/characters/snoopy_happy.png")}
    }

    cup_image = pygame.image.load("assets/emptycup.png").convert_alpha()

    BASES = {
        "milktea": {"image": pygame.image.load("assets/bases/milktea.png"), "value": 5},
        "matcha": {"image": pygame.image.load("assets/bases/matcha.png"), "value": 10},
        "taro": {"image": pygame.image.load("assets/bases/taro.png"), "value": 5}
    }

    TOPPINGS = {
        #"pearls": {"image": pygame.image.load("assets/toppings/pearls.png"), "value": 10},
        #"grassjelly": {"image": pygame.image.load("assets/toppings/grass_jelly.png"), "value": 15},
        #"coconut":{"image": pygame.image.load("assets/toppings/coconut.png"), "value": 10},
        #"strawberryboba":{"image": pygame.image.load("assets/toppings/strawberryboba.png"), "value": 10},
        #"mangoboba":{"image": pygame.image.load("assets/toppings/mangoboba.png"), "value": 10},
        #"strawberrychunk":{"image": pygame.image.load("assets/toppings/strawberry.png"), "value": 20},
        #"mangochunk":{"image": pygame.image.load("assets/toppings/mango.png"), "value": 20},
        #"redbean":{"image": pygame.image.load("assets/toppings/redbean.png"), "value": 20}
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
    HAND_TOPPINGS = ["pearls", "grassjelly", "coconut", "strawberryboba",
                     "mangoboba", "mangochunk", "strawberrychunk", "redbean"]

    # DEFAULT FONT (Sophia): falls back to pygame's font if the file is missing
    FONT_PATH = "assets/VT323-Regular.ttf"
    try:
        pygame.font.Font(FONT_PATH, 45)
    except (FileNotFoundError, OSError):
        print("Font not found:", FONT_PATH)
        FONT_PATH = None

    font = pygame.font.Font(FONT_PATH, 40)
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
        toppings_text = " and ".join(current_order) if current_order else "no toppings"
        return line.format(base=BASE_NAMES.get(current_base, current_base), toppings=toppings_text, topping=toppings_text)

    def get_mood():
        # happy = right order, meh = some toppings right, angry = none right
        if set(selected_toppings) == set(current_order):
            return "happy"
        else:
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

    # TITLE SCREEN (Sophia)
    def scale_to_width(image, width):
        height = int(image.get_height() * width / image.get_width())
        return pygame.transform.smoothscale(image, (width, height))

    def ease_out_back(t):
        # goes from 0 to 1, overshoots a little at the end, then settles (a small bounce)
        t = max(0, min(1, t))
        return 1 + 2.7 * (t - 1) ** 3 + 1.7 * (t - 1) ** 2

    def draw_button(text, center, lift, appear):
        # lift: 0 to 1 (hover), appear: 0 to 1 (load-in)
        size = 1 + 0.04 * lift
        image = pygame.transform.smoothscale(button_image, (int(button_image.get_width() * size), int(button_image.get_height() * size)))
        x = center[0] + int((1 - appear) * 300)      # slides in from the right
        y = center[1] - int(8 * lift)                # lifts up when hovered
        rect = image.get_rect(center=(x, y))
        alpha = int(255 * max(0, min(1, appear)))
        image.set_alpha(alpha)
        screen.blit(image, rect)
        label = button_font.render(text, True, (59, 38, 24))
        label.set_alpha(alpha)
        screen.blit(label, label.get_rect(center=(rect.centerx - 4, rect.centery - 4)))

    # STORY FLOW (Sophia): cutscene frames -> tutorial (placeholder) -> game
    def next_story_screen():
        nonlocal status, cutscene_index
        if status == "tutorial":
            if cutscene_index < len(cutscene) - 1:
                cutscene_index += 1
            else:
                status = "how_to_play"
        elif status == "how_to_play":
            status = "game"

    # KEY HINTS (Sophia): small floating note above something on the counter
    def draw_key_hint(letter, pos):
        bob = int(math.sin(pygame.time.get_ticks() / 300) * 3)    # gentle up-and-down float
        x, y = pos[0], pos[1] + bob
        note = pygame.Rect(0, 0, 36, 36)
        note.midbottom = (x, y - 8)
        pygame.draw.rect(screen, (255, 248, 235), note, border_radius=8)
        pygame.draw.rect(screen, (148, 90, 45), note, 2, border_radius=8)
        pygame.draw.polygon(screen, (148, 90, 45), [(x - 6, y - 9), (x + 6, y - 9), (x, y - 2)])
        label = font.render(letter, True, (148, 90, 45))
        screen.blit(label, label.get_rect(center=note.center))

    def generate_order():
        if len(available_toppings) == 0:
            return []    # no toppings set up yet: customers order just a base
        number = random.randint(1, len(available_toppings))
        return random.sample(available_toppings, number)

    def new_customer():
        nonlocal customer, current_order, current_base, selected_base

        name = random.choice(list(CUSTOMERS.keys()))
        current_base = random.choice(list(BASES.keys()))
        selected_base = None
        current_order = generate_order()
        customer = Customer(name, current_order, current_base)

        selected_toppings.clear()
        toppings.empty()

        positions = [
            (150, 450),
            (300, 450),
            (450, 450),
            (600, 450),
            (750, 450),
            (900, 450),
            (1050, 450),
            (1200, 450)
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

    def unlock_next_topping():
        nonlocal unlocked_toppings
        if unlocked_toppings < len(TOPPINGS):
            unlocked_toppings += 1
            available_toppings.append(TOPPINGS[unlocked_toppings - 1])
            print("New topping unlocked:", TOPPINGS[unlocked_toppings - 1])

    def clear_drink():
        nonlocal selected_base, cup_timer
        selected_base = None
        selected_toppings.clear()
        cup_timer = 350

    class Customer:
            def __init__(self, name, order, base):
                self.name = name
                self.order = order
                self.base = base
                self.image = CUSTOMERS[name]["default"]
                self.rect = self.image.get_rect(center=(850, 180))
                self.spawn_time = pygame.time.get_ticks() 
    
            def check_order(self, selected_toppings, selected_base):
                correct_toppings = set(selected_toppings) == set(self.order)
                correct_base = selected_base == self.base

                if correct_toppings and correct_base:
                    self.image = CUSTOMERS[self.name]["happy"]
                    return True
                else:
                    self.image = CUSTOMERS[self.name]["angry"]
                    return False
    
            def draw(self, screen):
                # POP-UP ANIMATION (Sophia): grows from the counter with a small bounce
                size = max(0.01, ease_out_back((pygame.time.get_ticks() - self.spawn_time) / 400))
                image = pygame.transform.smoothscale(self.image, (max(1, int(self.image.get_width() * size)), max(1, int(self.image.get_height() * size))))
                screen.blit(image, image.get_rect(midbottom=self.rect.midbottom))

    class Topping(pygame.sprite.Sprite):
        def __init__(self, image_path, x, y, name, value):
            super().__init__()
            self.image = image_path.convert_alpha()   # TOPPINGS already loads the image
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

    available_toppings = list(TOPPINGS.keys())[:3]
    unlocked_toppings = 3
    selected_toppings = []
    current_order = []
    customer = None

    customer_timer = 0
    next_customer_time = random.randint(500, 1500)

    customer_reacting = False
    reaction_timer = 0

    cutscene = [pygame.image.load("assets/ui/cutscene_frame1.png"), pygame.image.load("assets/ui/cutscene_frame2.png")]
    cutscene_index = 0

    # MAIN SCREEN BACKGROUND (Sophia)
    main_screen = pygame.transform.scale(pygame.image.load("assets/ui/background.png").convert(), (SCREEN_WIDTH, SCREEN_HEIGHT))
    camera_image = None

    # TITLE SCREEN (Sophia)
    title_image = scale_to_width(pygame.image.load("assets/ui/title.png").convert_alpha(), 735)
    button_image = scale_to_width(pygame.image.load("assets/ui/button.png").convert_alpha(), 350)
    dialogue_image = pygame.image.load("assets/ui/dialogue.png").convert_alpha()
    button_font = pygame.font.Font(FONT_PATH, 44)
    TITLE_CENTER = (410, 215)
    START_CENTER = (995, 185)
    QUIT_CENTER = (995, 340)
    start_rect = button_image.get_rect(center=START_CENTER)
    quit_rect = button_image.get_rect(center=QUIT_CENTER)
    title_timer = 0      # how long the title screen has been showing (ms)
    start_lift = 0       # hover animation, 0 = resting, 1 = fully lifted
    quit_lift = 0

    # EMPTY CUP (Sophia): press C on the game screen to place it on the cutting board
    CUP_POS = (555, 575)     # bottom-middle of the cup, on the cutting board
    cup_shown = False
    cup_timer = 0            # for the pop-up animation (ms)

    # BASES (Sophia): click a jar on the right of the background to fill the cup
    BASE_ZONES = {
        "taro": pygame.Rect(1150, 320, 120, 110),      # purple jar
        "matcha": pygame.Rect(1035, 455, 130, 135),    # green jar
        "milktea": pygame.Rect(1170, 430, 105, 160),   # brown jar
    }
    selected_base = None     # None = empty cup

    # KEY HINTS (Sophia): letter -> spot just above the thing on the counter
    KEY_HINTS = {
        "C": (1000, 315),    # cup stacks
        "I": (801, 382),     # ice bucket
    }
    
    tutorial_text = []

    # CUSTOMER DIALOGUE (Sophia)
    # names must match the keys in CUSTOMERS
    BASE_NAMES = {"milktea": "milk tea", "matcha": "matcha", "taro": "taro milk tea"}   # how bases are said out loud
    CUSTOMER_DIALOGUE = {
        "snoopy": {
            "order": ["A {base} with {toppings}... the perfect reward after a long day of napping."],
            "happy": ["*happy dance* ...This {base} is perfect. Best nap snack ever."],
            "meh": ["*tilts head* ...Not bad. Not great. Like a nap that was too short."],
            "angry": ["*flops over* ...ok."],
        },
        "hamster": {
            "order": ["HI CHAT CAN I GET A {base} WITH SOME {toppings} PLSPSLPSSPLS!!!!"],
            "happy": ["I LOVE THIS DRINK AND I LOVE YOUUUUUUU BYEEE"],
            "meh": ["Hmm, I think you made it wrong... that's okay ig."],
            "angry": ["Bro. Are u fr. I'm allergic to {base}"],
        },
        "redpanda": {
            "order": ["I'm hungry. Can I have a 67 ({base}) with some skibidi rizz ({toppings})"],
            "happy": ["Waiitttt imma touch u! It's perfect, thank youuu!"],
            "meh": ["Grr... it's okay I guess..."],
            "angry": ["GRRRRR this isnt what i asked for bro >:("],
        },
        "chiikawa": {
            "order": ["What do I want? Hmph.. it's not like i'd like a {base} with {toppings} or anything! And I definitely didn't come here just to see you... b-baka!"],
            "happy": ["Wow you made it right... but it's not like it was for me, s-so I don't care or anything!!!! H-hmph~"],
            "meh": ["Th-this is kind of close... b-but don't think I'm impressed or anything! Hmph."],
            "angry": ["b-baka! What the hell's wrong with you! Can't even make a simple drink..!"],
        },
        "peppapig": {
            "order": ["Two Two's my Poo Poo's, I finna take a {base} w some {toppings}, don't be a Gerbert fam."],
            "happy": ["Yooo this is fire! Bless up bossman!"],
            "meh": ["Nah who made this. I'm sooper cheesed fam."],
            "angry": ["cro... wrong order fam."],
        },
    }
    dialogue_font = pygame.font.Font(FONT_PATH, 36)
    dialogue_box = pygame.Rect(30, 520, 950, 180)     # bottom of the screen, left of the SERVE button
    dialogue_close_button = pygame.Rect(900, 525, 65, 35)
    dialogue_visible = True
    dialogue_customer = None     # the customer the current dialogue belongs to
    dialogue_text = ""
    current_base = None
    reaction_given = False

    serve_button = pygame.Rect(980, 600, 200, 70)
    clear_button = pygame.Rect(760, 600, 200, 70)

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
                elif (event.key == K_RETURN or event.key == K_SPACE) and status in ("tutorial", "how_to_play"):
                    next_story_screen()
                elif event.key == K_c and status == "game":
                    # EMPTY CUP (Sophia)
                    cup_shown = True
                    cup_timer = 0
            elif event.type == MOUSEBUTTONDOWN:
                if status == "mode_select":
                    if browser_button.collidepoint(event.pos):
                        status = "title"
                    elif download_button.collidepoint(event.pos):
                        start_camera()
                        status = "title"
                elif status in ("tutorial", "how_to_play"):
                    # STORY FLOW (Sophia): click anywhere to go to the next screen
                    next_story_screen()
                elif status == "title":
                    # TITLE SCREEN (Sophia)
                    if title_timer > 1400:
                        if start_rect.collidepoint(event.pos):
                            status = "tutorial"
                        elif quit_rect.collidepoint(event.pos):
                            running = False
                elif status == "game":
                    if dialogue_text != "" and dialogue_close_button.collidepoint(event.pos):
                        dialogue_visible = False
                    if not use_camera and customer is not None and not customer_reacting:
                            clicked = False

                            for topping in reversed(toppings.sprites()):
                                if topping.rect.collidepoint(event.pos) and not clicked:
                                    topping.handle_click()
                                    clicked = True

                            if serve_button.collidepoint(event.pos):
                                accepted = customer.check_order(selected_toppings, selected_base)

                                customer_reacting = True
                                reaction_timer = 0

                                if accepted:
                                    money += BASES[current_base]["value"]
                                    money += sum(
                                        TOPPINGS[name]["value"]
                                        for name in current_order
                                        )
                                    unlock_next_topping()

                    # OPENCV MODE (Sophia): serve is still done with the mouse
                    if use_camera and customer is not None and not customer_reacting:
                        if serve_button.collidepoint(event.pos):
                            accepted = customer.check_order(selected_toppings, selected_base)

                            customer_reacting = True
                            reaction_timer = 0

                            if accepted:
                                money += BASES[current_base]["value"]
                                money += sum(
                                    TOPPINGS[name]["value"]
                                    for name in current_order
                                    )
                                unlock_next_topping()

                    # BASES (Sophia): click a jar to fill the cup (cup has to be out first, press C)
                    if cup_shown:
                        for base_name, zone in BASE_ZONES.items():
                            if zone.collidepoint(event.pos):
                                selected_base = base_name

                    if clear_button.collidepoint(event.pos):
                        if customer is not None and not customer_reacting and cup_shown:
                            clear_drink()

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
            # TITLE SCREEN
            title_timer += dt

            # background fades in from black
            main_screen.set_alpha(min(255, int(255 * title_timer / 600)))
            screen.blit(main_screen, (0, 0))
            main_screen.set_alpha(255)

            # title drops in from above with a small bounce
            t = (title_timer - 400) / 700
            appear = ease_out_back(t)
            title_image.set_alpha(int(255 * max(0, min(1, t))))
            screen.blit(title_image, title_image.get_rect(center=(TITLE_CENTER[0], TITLE_CENTER[1] - int((1 - appear) * 120))))

            mouse_pos = pygame.mouse.get_pos()
            start_target = 1 if start_rect.collidepoint(mouse_pos) else 0
            quit_target = 1 if quit_rect.collidepoint(mouse_pos) else 0
            start_lift += (start_target - start_lift) * 0.18
            quit_lift += (quit_target - quit_lift) * 0.18

            # buttons slide in from the right
            draw_button("start", START_CENTER, start_lift, ease_out_back((title_timer - 800) / 600))
            draw_button("quit", QUIT_CENTER, quit_lift, ease_out_back((title_timer - 950) / 600))

            if mode_message != "":
                message_text = font.render(mode_message, True, (255, 255, 255))
                screen.blit(message_text, message_text.get_rect(center=(640, 650)))
            #screen.blit(title_image,(0,0))
            pass
        elif status == "tutorial":
            screen.blit(cutscene[cutscene_index], (0, 0))
        elif status == "how_to_play":
            # TUTORIAL SPOT (Sophia): black screen for now, put the tutorial here later
            hint_text = font.render("click to start", True, (90, 90, 90))
            screen.blit(hint_text, hint_text.get_rect(center=(640, 680)))
        elif status == "game":
            # MAIN SCREEN BACKGROUND
            screen.blit(main_screen, (0, 0))
            if use_camera and camera_image is not None:
                # small webcam preview so players can still see their hands
                screen.blit(pygame.transform.scale(camera_image, (320, 240)), (SCREEN_WIDTH - 330, 10))

            # EMPTY CUP (Sophia): pops up with a small bounce
            if cup_shown:
                cup_timer += dt
                size = max(0.01, ease_out_back(cup_timer / 350))
                cup_picture = cup_image if selected_base is None else BASES[selected_base]["image"]
                cup = pygame.transform.smoothscale(cup_picture, (max(1, int(cup_picture.get_width() * size)), max(1, int(cup_picture.get_height() * size))))
                screen.blit(cup, cup.get_rect(midbottom=CUP_POS))

            # BASES (Sophia): hover highlight on the jars (temporary look)
            mouse_pos = pygame.mouse.get_pos()
            for base_name, zone in BASE_ZONES.items():
                if zone.collidepoint(mouse_pos):
                    highlight = pygame.Surface(zone.size, pygame.SRCALPHA)
                    pygame.draw.rect(highlight, (255, 255, 255, 70), highlight.get_rect(), border_radius=14)
                    screen.blit(highlight, zone)
                    pygame.draw.rect(screen, (148, 90, 45), zone, 3, border_radius=14)
                    base_label = font.render(base_name, True, (148, 90, 45))
                    screen.blit(base_label, base_label.get_rect(midbottom=(zone.centerx, zone.top - 4)))

            # KEY HINTS (Sophia)
            for letter, pos in KEY_HINTS.items():
                draw_key_hint(letter, pos)

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
                        #new customer wait time
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

                #CLEAR BUTTON
                pygame.draw.rect(screen, (180, 90, 90), clear_button, border_radius=10)
                pygame.draw.rect(screen, (255, 255, 255), clear_button, 2, border_radius=10)
                clear_text = font.render("CLEAR", True, (255, 255, 255))
                screen.blit(clear_text, clear_text.get_rect(center=clear_button.center))

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
                        #new customer wait time
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

                # CLEAR BUTTON
                pygame.draw.rect(screen, (180, 90, 90), clear_button, border_radius=10)
                pygame.draw.rect(screen, (255, 255, 255), clear_button, 2, border_radius=10)
                clear_text = font.render("CLEAR DRINK", True, (255, 255, 255))
                screen.blit(clear_text, clear_text.get_rect(center=clear_button.center))

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

                reaction_given = False
                dialogue_visible = True
                dialogue_text = ""

                if customer.name in CUSTOMER_DIALOGUE:
                    dialogue_text = fill_in_dialogue(
                        random.choice(
                            CUSTOMER_DIALOGUE[customer.name]["order"]
                        )
                    )

            # drink was served: switch to a reaction line
            
            if customer is not None and customer_reacting and not reaction_given:
                reaction_given = True
                dialogue_visible = True
                if customer.name in CUSTOMER_DIALOGUE:
                    # Check whether BOTH base and toppings were correct
                    correct_toppings = set(selected_toppings) == set(customer.order)
                    correct_base = selected_base == customer.base

                    if correct_toppings and correct_base:
                        mood = "happy"

                    elif any(name in customer.order for name in selected_toppings) and correct_base:
                        mood = "meh"

                    else:
                        mood = "angry"

                    dialogue_text = fill_in_dialogue(
                        random.choice(CUSTOMER_DIALOGUE[customer.name][mood])
                    )


            # customer left
            if customer is None:
                dialogue_text = ""
                dialogue_visible = True

            # DRAW DIALOGUE
            if dialogue_text != "" and dialogue_visible:
                screen.blit(
                    pygame.transform.smoothscale(dialogue_image, dialogue_box.size),
                    dialogue_box
                )

                y = dialogue_box.top + 30
                for line in wrap_text(dialogue_text, dialogue_font, dialogue_box.width - 100):
                    line_surface = dialogue_font.render(line, True, (0, 0, 0))
                    screen.blit(line_surface, (dialogue_box.left + 40, y))
                    y += dialogue_font.get_linesize()

                    # CLOSE BUTTON
                pygame.draw.rect(screen, (180, 90, 90), dialogue_close_button, border_radius=8)
                pygame.draw.rect(screen, (255, 255, 255), dialogue_close_button, 2, border_radius=8)
                close_text = font.render("X", True, (255, 255, 255))
                screen.blit(close_text, close_text.get_rect(center=dialogue_close_button.center))

            money_text = font.render(f"Money: ${money}", True, (148, 90, 45))
            screen.blit(money_text, (20, 20))

            # WHEN ADDING MORE TOPPINGS:
            # toppings.add(Topping("assets/toppings/example.png", 700, 300, "example", 20))

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    if camera is not None:
        camera.release()
    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())