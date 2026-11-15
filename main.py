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
        "pearls": {"image": pygame.image.load("assets/toppings/pearls.png"), "value": 10},
        "grassjelly": {"image": pygame.image.load("assets/toppings/grassjelly.png"), "value": 15},
        "coconutjelly":{"image": pygame.image.load("assets/toppings/coconutjelly.png"), "value": 10},
        "strawberrypearls":{"image": pygame.image.load("assets/toppings/strawberrypearls.png"), "value": 10},
        "mangopearls":{"image": pygame.image.load("assets/toppings/mangopearls.png"), "value": 10},
        "strawberrychunks":{"image": pygame.image.load("assets/toppings/strawberrychunks.png"), "value": 20},
        "mangochunks":{"image": pygame.image.load("assets/toppings/mangochunks.png"), "value": 20},
        "redbean":{"image": pygame.image.load("assets/toppings/redbean.png"), "value": 20}
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
    MAX_HANDS = 1           # 1 = one hand (1-5 fingers, more reliable), 2 = both hands (1-8)
    HOLD_TIME = 15          # frames to hold the same number (about 0.25-0.5 sec, depends on the camera's speed)
    waiting_for_reset = False
    hand_points = []        # the 21 points of each hand found this frame (for drawing)
    camera_size = (640, 480)
    # which points to join with lines to draw the hand (wrist, then each finger)
    HAND_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                  (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

    # finger number -> topping. Names must match the keys in TOPPINGS
    HAND_TOPPINGS = ["pearls", "grassjelly", "coconutjelly", "strawberrypearls",
                     "mangopearls", "mangochunks", "strawberrychunks", "redbean"]

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
        nonlocal hand_points, camera_size
        total = 0
        hand_points = []
        camera_size = (frame.shape[1], frame.shape[0])
        palms = palm_detector.infer(frame)
        for palm in palms[:MAX_HANDS]:               # only look at MAX_HANDS hands
            hand = hand_detector.infer(frame, palm)
            if hand is not None:
                points = hand[4:67].reshape(21, 3)    # 21 points, each (x, y, z)
                hand_points.append(points)
                total += count_fingers_on_hand(points)
        return total

    def draw_hand_preview(preview=None, show_info=True):
        # webcam preview (left side, under the money, by default) with the tracked hands drawn on top
        # kept on the left so it never covers the cups, the C note or the tea jars on the right
        if preview is None:
            preview = pygame.Rect(20, 104, 256, 192)
        pygame.draw.rect(screen, (148, 90, 45), preview.inflate(10, 10), border_radius=12)
        screen.blit(pygame.transform.scale(camera_image, preview.size), preview)
        scale_x = preview.width / camera_size[0]
        scale_y = preview.height / camera_size[1]
        for points in hand_points:
            spots = [(preview.left + int(p[0] * scale_x), preview.top + int(p[1] * scale_y)) for p in points]
            for a, b in HAND_LINES:
                pygame.draw.line(screen, (255, 244, 220), spots[a], spots[b], 3)
            for spot in spots:
                pygame.draw.circle(screen, (122, 150, 112), spot, 5)
                pygame.draw.circle(screen, (255, 244, 220), spot, 5, 1)

        # finger count badge on the preview
        badge = pygame.Rect(preview.left + 10, preview.bottom - 58, 58, 48)
        pygame.draw.rect(screen, SAGE, badge, border_radius=10)
        number = money_font.render(str(finger_count), True, (255, 244, 220))
        screen.blit(number, number.get_rect(center=badge.center))
        if len(hand_points) == 0:
            hint = key_font.render("show your hands", True, (255, 244, 220))
            screen.blit(hint, hint.get_rect(midleft=(badge.right + 10, badge.centery)))

        if not show_info:
            return

        # what the number picks + hold progress bar, under the preview
        info = pygame.Rect(preview.left - 5, preview.bottom + 12, 430, 60)
        pygame.draw.rect(screen, (255, 250, 238), info, border_radius=10)
        pygame.draw.rect(screen, (148, 90, 45), info, 3, border_radius=10)
        if picking_toppings():
            if 1 <= finger_count <= len(HAND_TOPPINGS):
                name = HAND_TOPPINGS[finger_count - 1]
                locked = name not in available_toppings
                text = str(finger_count) + " = " + TOPPING_NAMES[name] + (" (locked)" if locked else "")
            else:
                text = "hold up 1-" + str(5 if MAX_HANDS == 1 else 8) + " fingers"
            if waiting_for_reset:
                text = "added! put your hands down"
        elif customer is not None and not customer_reacting:
            text = "toppings done! serve [S]"
        else:
            text = "waiting for a customer..."
        label = tag_font.render(text, True, (110, 62, 30))
        screen.blit(label, label.get_rect(midtop=(info.centerx, info.top + 8)))
        bar = pygame.Rect(info.left + 14, info.bottom - 22, info.width - 28, 12)
        pygame.draw.rect(screen, (230, 215, 190), bar, border_radius=6)
        if picking_toppings() and hold_frames > 0:
            filled = bar.copy()
            filled.width = int(bar.width * min(1, hold_frames / HOLD_TIME))
            pygame.draw.rect(screen, (122, 150, 112), filled, border_radius=6)

    def add_topping(name):
        # adds a topping to the cup (in the order you pick them), max 3, only unlocked ones
        if name in available_toppings and name not in selected_toppings and len(selected_toppings) < 3:
            selected_toppings.append(name)
            topping_drop_start[name] = pygame.time.get_ticks()   # starts the drop-in bounce

    def add_topping_by_hand(name):
        add_topping(name)

    def picking_toppings():
        # hand motions only work while a customer is waiting and the toppings aren't done
        if customer is None or customer_reacting:
            return False
        if len(selected_toppings) >= len(current_order) or len(selected_toppings) >= 3:
            return False
        return True

    # CUSTOMER DIALOGUE (Sophia)
    def nice_list(names):
        # ["pearls", "redbean", "mangochunks"] -> "pearls, red bean and mango chunks"
        words = [TOPPING_NAMES.get(name, name) for name in names]
        if len(words) == 0:
            return "no toppings"
        if len(words) == 1:
            return words[0]
        return ", ".join(words[:-1]) + " and " + words[-1]

    def fill_in_dialogue(line):
        toppings_text = nice_list(current_order)
        return line.format(base=BASE_NAMES.get(current_base, current_base), toppings=toppings_text, topping=toppings_text)

    def get_mood():
        # happy = right base + right toppings in the right order
        # meh   = right base, but toppings in the wrong order (or only some toppings right)
        # angry = wrong base, or none of the toppings right
        # ice: getting ice wrong turns a perfect drink into "meh"
        correct_base = selected_base == current_base
        correct_ice = ice_added == current_ice
        if correct_base and selected_toppings == current_order:
            return "happy" if correct_ice else "meh"
        if correct_base and set(selected_toppings) == set(current_order):
            return "meh"
        if correct_base and any(name in current_order for name in selected_toppings):
            return "meh"
        return "angry"

    def order_total():
        # what the customer pays for a perfect drink
        return BASES[current_base]["value"] + sum(TOPPINGS[name]["value"] for name in current_order)

    def serve_drink():
        # MONEY ONLY CHANGES HERE: happy = full price, meh = half, angry = you lose half (wasted ingredients)
        nonlocal money, customer_reacting, reaction_timer, last_mood, money_popup_text, money_popup_timer
        nonlocal cup_shown, selected_base, served_base, served_toppings, served_timer
        nonlocal ice_added, served_ice
        last_mood = get_mood()
        total = order_total()
        if last_mood == "happy":
            change = total
            unlock_next_topping()
        elif last_mood == "meh":
            change = total // 2
        else:
            change = -(total // 2)
        money += change
        money_popup_text = ("+$" if change >= 0 else "-$") + str(abs(change))
        money_popup_timer = 0
        customer.image = CUSTOMERS[customer.name].get(last_mood, CUSTOMERS[customer.name]["default"])
        customer_reacting = True
        reaction_timer = 0
        # the drink is handed over: it leaves the counter, press C for a new cup
        served_base = selected_base
        served_toppings = list(selected_toppings)
        served_ice = ice_added
        served_timer = 0
        cup_shown = False
        selected_base = None
        selected_toppings.clear()
        ice_added = False

    def draw_ui_button(name, rect, text, key_text, fill, text_colour, mouse_pos):
        # rounded button with a drop shadow, lifts and brightens a little on hover
        target = 1 if rect.collidepoint(mouse_pos) else 0
        button_hover[name] = button_hover.get(name, 0) + (target - button_hover.get(name, 0)) * 0.2
        lift = int(button_hover[name] * 4)
        shadow = rect.move(0, 6)
        darker = (max(0, fill[0] - 55), max(0, fill[1] - 55), max(0, fill[2] - 55))
        pygame.draw.rect(screen, darker, shadow, border_radius=14)
        face = rect.move(0, -lift)
        lighter = tuple(min(255, int(c + (255 - c) * 0.15 * button_hover[name])) for c in fill)
        pygame.draw.rect(screen, lighter, face, border_radius=14)
        pygame.draw.rect(screen, darker, face, 3, border_radius=14)
        # thin highlight along the top edge
        pygame.draw.line(screen, tuple(min(255, c + 45) for c in lighter), (face.left + 14, face.top + 5), (face.right - 14, face.top + 5), 2)
        label = font.render(text, True, text_colour)
        key_label = key_font.render(key_text, True, text_colour)
        key_label.set_alpha(190)
        total_width = label.get_width() + 8 + key_label.get_width()
        x = face.centerx - total_width // 2
        screen.blit(label, label.get_rect(midleft=(x, face.centery)))
        screen.blit(key_label, key_label.get_rect(midleft=(x + label.get_width() + 8, face.centery + 2)))

    def draw_money():
        # big sage-green money box in the top left
        panel = pygame.Rect(20, 16, 250, 76)
        pygame.draw.rect(screen, (245, 240, 225), panel, border_radius=14)
        pygame.draw.rect(screen, SAGE, panel, 4, border_radius=14)
        sign_box = pygame.Rect(panel.left + 12, panel.top + 12, 52, 52)
        pygame.draw.rect(screen, SAGE, sign_box, border_radius=10)
        sign = money_font.render("$", True, (245, 240, 225))
        screen.blit(sign, sign.get_rect(center=sign_box.center))
        shown = int(round(displayed_money))      # counts up/down smoothly toward the real money
        amount_colour = SAGE if shown >= 0 else (150, 60, 50)
        amount = money_font.render(str(shown), True, amount_colour)
        screen.blit(amount, amount.get_rect(midleft=(sign_box.right + 14, panel.centery)))
        # +$ / -$ floats up and fades after serving
        if money_popup_text != "" and money_popup_timer < 1300:
            popup_colour = SAGE if money_popup_text.startswith("+") else (150, 60, 50)
            popup = font.render(money_popup_text, True, popup_colour)
            popup.set_alpha(int(255 * (1 - money_popup_timer / 1300)))
            screen.blit(popup, popup.get_rect(midleft=(panel.right + 14, panel.centery - int(money_popup_timer / 40))))

    def draw_topping_buttons(mouse_pos):
        # soft hover buttons over the 4 x 2 topping jars on the background
        for name, zone in TOPPING_ZONES.items():
            target = 1 if zone.collidepoint(mouse_pos) else 0
            topping_hover[name] += (target - topping_hover[name]) * 0.2
            amount = topping_hover[name]
            if amount < 0.02:
                continue
            unlocked = name in available_toppings
            glow = pygame.Surface(zone.size, pygame.SRCALPHA)
            fill = (255, 250, 238, int(110 * amount)) if unlocked else (60, 60, 60, int(110 * amount))
            pygame.draw.rect(glow, fill, glow.get_rect(), border_radius=12)
            screen.blit(glow, zone)
            border = pygame.Surface(zone.size, pygame.SRCALPHA)
            border_colour = (148, 90, 45, int(255 * amount)) if unlocked else (90, 90, 90, int(255 * amount))
            pygame.draw.rect(border, border_colour, border.get_rect(), 3, border_radius=12)
            screen.blit(border, zone)
            # name tag above the jar
            text = TOPPING_NAMES[name] if unlocked else "locked"
            label = tag_font.render(text, True, (255, 250, 238))
            tag = label.get_rect(midbottom=(zone.centerx, zone.top - 6 - int(4 * amount))).inflate(16, 8)
            tag_surface = pygame.Surface(tag.size, pygame.SRCALPHA)
            tag_colour = (148, 90, 45, int(235 * amount)) if unlocked else (90, 90, 90, int(235 * amount))
            pygame.draw.rect(tag_surface, tag_colour, tag_surface.get_rect(), border_radius=8)
            screen.blit(tag_surface, tag)
            label.set_alpha(int(255 * amount))
            screen.blit(label, label.get_rect(center=tag.center))

    def ease_out_bounce(t):
        # falls and bounces a couple of times before settling (like a ball landing)
        t = max(0, min(1, t))
        if t < 1 / 2.75:
            return 7.5625 * t * t
        elif t < 2 / 2.75:
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < 2.5 / 2.75:
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375

    def draw_cup(base, cup_toppings, bottom_middle, size, alpha, ice=False):
        # one cup with its base and toppings, plus a soft shadow so it stands out from the board
        picture = cup_image if base is None else BASES[base]["image"]
        width = max(1, int(picture.get_width() * size))
        height = max(1, int(picture.get_height() * size))
        shadow = pygame.Surface((int(width * 0.9) + 2, 22), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (60, 35, 20, int(70 * alpha / 255)), shadow.get_rect())
        screen.blit(shadow, shadow.get_rect(center=(bottom_middle[0], bottom_middle[1] - 2)))
        layers = [(picture, 0)]
        for name in cup_toppings:
            # a topping that was just added drops in from above and bounces into place
            drop = 0
            if name in topping_drop_start:
                t = (pygame.time.get_ticks() - topping_drop_start[name]) / 600
                drop = int((1 - ease_out_bounce(t)) * 110)
            layers.append((TOPPINGS[name]["image"], drop))
        if ice:
            # ice drops in last, on top, with the same bounce
            t = (pygame.time.get_ticks() - ice_drop_start) / 600
            layers.append((ice_image, int((1 - ease_out_bounce(t)) * 110)))
        for image, drop in layers:
            scaled = pygame.transform.smoothscale(image, (max(1, int(image.get_width() * size)), max(1, int(image.get_height() * size))))
            scaled.set_alpha(alpha)
            screen.blit(scaled, scaled.get_rect(midbottom=(bottom_middle[0], bottom_middle[1] - drop)))

    def get_cup():
        # C key or clicking the cup stacks
        nonlocal cup_shown, cup_timer
        if not cup_shown:
            cup_shown = True
            cup_timer = 0

    def toggle_ice():
        # I key or clicking the ice bucket: adds ice, or takes it back out if it's already in
        nonlocal ice_added, ice_drop_start
        if cup_shown:
            ice_added = not ice_added
            ice_drop_start = pygame.time.get_ticks()

    def draw_counter_highlight(zone, text, mouse_pos):
        # same soft highlight as the base jars, with the name + key above
        if zone.collidepoint(mouse_pos):
            highlight = pygame.Surface(zone.size, pygame.SRCALPHA)
            pygame.draw.rect(highlight, (255, 255, 255, 70), highlight.get_rect(), border_radius=14)
            screen.blit(highlight, zone)
            pygame.draw.rect(screen, (148, 90, 45), zone, 3, border_radius=14)
            label = tag_font.render(text, True, (255, 250, 238))
            tag = label.get_rect(midbottom=(zone.centerx, zone.top - 8)).inflate(16, 8)
            pygame.draw.rect(screen, (148, 90, 45), tag, border_radius=8)
            screen.blit(label, label.get_rect(center=tag.center))

    def dialogue_done_typing():
        return dialogue_timer * TYPE_SPEED >= len(dialogue_text)

    def advance_dialogue():
        # click / Enter: first finish the typing, then close the box
        nonlocal dialogue_timer, dialogue_visible
        if not dialogue_done_typing():
            dialogue_timer = len(dialogue_text) / TYPE_SPEED + 1000
        else:
            dialogue_visible = False

    def draw_dialogue():
        # everything is drawn on its own layer so the whole box can fade in and out together
        layer = pygame.Surface((dialogue_box.width, dialogue_box.height + 44), pygame.SRCALPHA)
        box = pygame.Rect(0, 44, dialogue_box.width, dialogue_box.height)
        layer.blit(pygame.transform.smoothscale(dialogue_image, box.size), box)

        # name tab on the top left
        if customer is not None:
            name = CUSTOMER_DISPLAY_NAMES.get(customer.name, customer.name.title())
            name_surface = name_font.render(name, True, (255, 244, 220))
            tab = pygame.Rect(28, 4, name_surface.get_width() + 44, 48)
            pygame.draw.rect(layer, (148, 90, 45), tab, border_radius=12)
            pygame.draw.rect(layer, (110, 62, 30), tab, 3, border_radius=12)
            layer.blit(name_surface, name_surface.get_rect(center=tab.center))

        # typewriter text: letters appear one by one, already wrapped so words don't jump lines
        letters_shown = int(dialogue_timer * TYPE_SPEED)
        y = box.top + 30
        for line in wrap_text(dialogue_text, dialogue_font, box.width - 140):
            if letters_shown <= 0:
                break
            part = line[:letters_shown]
            letters_shown -= len(line) + 1
            layer.blit(dialogue_font.render(part, True, (45, 30, 20)), (box.left + 44, y))
            y += dialogue_font.get_linesize()

        # X button (top right of the box)
        close = dialogue_close_button.move(-dialogue_box.left, -dialogue_box.top + 44)
        pygame.draw.rect(layer, (180, 90, 90), close, border_radius=8)
        pygame.draw.rect(layer, (255, 255, 255), close, 2, border_radius=8)
        x_text = font.render("X", True, (255, 255, 255))
        layer.blit(x_text, x_text.get_rect(center=close.center))

        # little arrow that bobs up and down once the text is done
        if dialogue_done_typing():
            bob = math.sin(pygame.time.get_ticks() / 220) * 5
            ax, ay = box.right - 48, box.bottom - 34 + bob
            pygame.draw.polygon(layer, (148, 90, 45), [(ax - 12, ay - 8), (ax + 12, ay - 8), (ax, ay + 8)])

        # pop up slowly with the same bounce as the customers, and fade in / out
        appear = ease_out_back(dialogue_timer / 650)
        alpha = min(dialogue_fade, max(0, min(1, dialogue_timer / 300)))
        layer.set_alpha(int(255 * alpha))
        screen.blit(layer, (dialogue_box.left, dialogue_box.top - 44 + int((1 - appear) * 50)))

    def draw_story_box(text, shake):
        # same look as the customer dialogue: full-width box, name tab, typewriter text, bobbing arrow
        layer = pygame.Surface((dialogue_box.width, dialogue_box.height + 44), pygame.SRCALPHA)
        box = pygame.Rect(0, 44, dialogue_box.width, dialogue_box.height)
        layer.blit(pygame.transform.smoothscale(dialogue_image, box.size), box)

        name_surface = name_font.render(STORY_NAME, True, (255, 244, 220))
        tab = pygame.Rect(28, 4, name_surface.get_width() + 44, 48)
        pygame.draw.rect(layer, (148, 90, 45), tab, border_radius=12)
        pygame.draw.rect(layer, (110, 62, 30), tab, 3, border_radius=12)
        layer.blit(name_surface, name_surface.get_rect(center=tab.center))

        letters_shown = int(max(0, story_timer) * TYPE_SPEED)
        y = box.top + 30
        for line in wrap_text(text, dialogue_font, box.width - 140):
            if letters_shown <= 0:
                break
            part = line[:letters_shown]
            letters_shown -= len(line) + 1
            # shaky lines: every letter jitters a little
            if shake:
                x = box.left + 44
                for letter in part:
                    letter_surface = dialogue_font.render(letter, True, (45, 30, 20))
                    layer.blit(letter_surface, (x + random.randint(-2, 2), y + random.randint(-2, 2)))
                    x += dialogue_font.size(letter)[0]
            else:
                layer.blit(dialogue_font.render(part, True, (45, 30, 20)), (box.left + 44, y))
            y += dialogue_font.get_linesize()

        if story_timer * TYPE_SPEED >= len(text):
            bob = math.sin(pygame.time.get_ticks() / 220) * 5
            ax, ay = box.right - 48, box.bottom - 34 + bob
            pygame.draw.polygon(layer, (148, 90, 45), [(ax - 12, ay - 8), (ax + 12, ay - 8), (ax, ay + 8)])

        # pops up with a slow bounce and fades in at the start of each picture
        appear = ease_out_back(story_frame_timer / 650)
        layer.set_alpha(int(255 * max(0, min(1, story_frame_timer / 300))))
        shake_x, shake_y = 0, 0
        if shake and story_timer > 0:
            strength = 7 if story_timer < 700 else 3        # big shake at first, then keeps trembling
            shake_x, shake_y = random.randint(-strength, strength), random.randint(-strength, strength)
        screen.blit(layer, (dialogue_box.left + shake_x, dialogue_box.top - 44 + int((1 - appear) * 50) + shake_y))

    def draw_receipt():
        # the order, top right, so you remember it after closing the dialogue
        lines = [("------ Base ------", True), (BASE_NAMES[current_base], False), ("---- Toppings ----", True)]
        if current_order:
            for number, name in enumerate(current_order, 1):
                lines.append((str(number) + ". " + TOPPING_NAMES[name], False))
        else:
            lines.append(("none", False))
        lines.append(("------- Ice -------", True))
        lines.append(("with ice" if current_ice else "no ice", False))
        lines.append(("------ Total ------", True))
        lines.append(("$" + str(order_total()), False))
        line_height = receipt_font.get_linesize()
        width = 280
        height = 30 + line_height * len(lines)
        if use_camera:
            paper = pygame.Rect(290, 16, width, height)                     # next to the money box in hand mode
        else:
            paper = pygame.Rect(SCREEN_WIDTH - width - 20, 20, width, height)  # top right in clicking mode
        pygame.draw.rect(screen, (255, 250, 238), paper, border_radius=6)
        pygame.draw.rect(screen, (148, 90, 45), paper, 3, border_radius=6)
        y = paper.top + 15
        for text, is_heading in lines:
            colour = (148, 90, 45) if is_heading else (40, 30, 20)
            line_surface = receipt_font.render(text, True, colour)
            screen.blit(line_surface, line_surface.get_rect(midtop=(paper.centerx, y)))
            y += line_height

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

    # STORY FLOW (Sophia): cutscene frames -> tutorial -> game

    def story_line():
        return STORY_LINES[cutscene_index][tutorial_text_index]

    def next_story_screen():
        nonlocal status, cutscene_index, tutorial_text_index, story_timer, story_frame_timer
        if status == "tutorial":
            if cutscene_index in STORY_LINES:
                # STORY: first finish typing, then the next line, then the next frame
                text = story_line()[0]
                if story_timer * TYPE_SPEED < len(text):
                    story_timer = len(text) / TYPE_SPEED + 1000
                elif tutorial_text_index < len(STORY_LINES[cutscene_index]) - 1:
                    tutorial_text_index += 1
                    story_timer = 0
                else:
                    tutorial_text_index = 0
                    cutscene_index += 1
                    story_timer = -450          # wait for the picture to fade in first
                    story_frame_timer = -450
                    if cutscene_index >= len(cutscene):
                        status = "how_to_play"
            elif cutscene_index in TUTORIAL_INSTRUCTIONS:
                if tutorial_text_index < len(TUTORIAL_INSTRUCTIONS[cutscene_index]) - 1:
                    tutorial_text_index += 1
                else:
                    tutorial_text_index = 0
                    cutscene_index += 1
                    if cutscene_index >= len(cutscene):
                        status = "how_to_play"
            else:
                cutscene_index += 1
                tutorial_text_index = 0
                if cutscene_index >= len(cutscene):
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
        number = random.randint(1, min(3, len(available_toppings)))
        return random.sample(available_toppings, number)    # the order of this list matters

    def new_customer():
        nonlocal customer, current_order, current_base, selected_base, current_ice

        name = random.choice(list(CUSTOMERS.keys()))
        current_base = random.choice(list(BASES.keys()))
        current_ice = random.choice([True, False])     # ICE (Sophia): with ice or no ice
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

    def draw_tutorial_text():
        instruction = TUTORIAL_INSTRUCTIONS[cutscene_index][tutorial_text_index]
        pygame.draw.rect(screen, (255, 248, 235), tutorial_box, border_radius=12)
        pygame.draw.rect(screen, (148, 90, 45), tutorial_box, 3, border_radius=12)
        lines = wrap_text(instruction, tutorial_font, tutorial_box.width - 30)
        y = tutorial_box.top + 20
        for line in lines:
            text_surface = tutorial_font.render(line, True, (75, 50, 35))
            screen.blit(text_surface, (tutorial_box.left + 15, y))
            y += tutorial_font.get_linesize()
            hint = tutorial_hint_font.render("SPACE to continue", True, (148, 90, 45))
            screen.blit(hint, hint.get_rect(bottomright=(tutorial_box.right - 12, tutorial_box.bottom - 10)))

    def unlock_next_topping():
        nonlocal unlocked_toppings
        if unlocked_toppings < len(TOPPINGS):
            unlocked_toppings += 1
            new_topping = list(TOPPINGS.keys())[unlocked_toppings - 1]
            available_toppings.append(new_topping)
            print("New topping unlocked:", new_topping)

    def clear_drink():
        nonlocal selected_base, cup_timer, ice_added
        selected_base = None
        ice_added = False
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

    cutscene = [pygame.image.load("assets/ui/cutscene_frame1.png"), pygame.image.load("assets/ui/cutscene_frame2.png"), pygame.image.load("assets/ui/cutscene_frame3.png"), pygame.image.load("assets/ui/tutorial1.png"), pygame.image.load("assets/ui/tutorial2.png")]
    cutscene_index = 0
    tutorial_text_index = 0

    # STORY (Sophia): visual-novel lines for the first 3 cutscene frames. (text, shake?)
    STORY_NAME = "You"
    STORY_LINES = {
        0: [
            ("After resigning from my position as CEO of Nvidia 6 months ago, my life really sucks. I've just been bedrotting, I don't even wanna get up. I miss my wife and 3 girlfriends. Everyone calls me funky unc. My life is ruined...", False),
            ("And worst of all...", False),
        ],
        1: [
            ("I'M BROKE!!! I guess I'll have to sell my 24k gold labubu.", True),
        ],
        2: [
            ("I can't even afford my favourite bubble tea that I used to get my girlfriend(s) anymore. That was the only way I could remember them... hmm, if I can't afford to buy it, I can always make my own.", False),
        ],
    }
    story_timer = -450        # ms since the current line started (negative = wait for the screen fade)
    story_frame_timer = -450  # ms since the current frame started (for the box popping in)

    TUTORIAL_INSTRUCTIONS = {
        3: [
            "Welcome to Boba Comeback, where you attempt to bounce back financially after going horribly bankrupt!",
            "Customers, like Peppa Pig over here, will come in and order drinks from you.",
            "It is your job to correctly make them. The order matters!"
        ],
        4: [
            "Click C to place a cup on the center board.",
            "Then, click any of the three jars on the right to choose your base.",
            "Click the left topping boxes to add the specified toppings into the cup.",
            "You'll notice that you only have three toppings unlocked at the beginning.",
            "As you make more drinks, you'll unlock more toppings.",
            "Press the CLEAR button to reset your cup, and press the SERVE button to give the customer the drink.",
            "Make 100$ to fully bounce back from being so incredibly broke!"
        ]
    }

    # MAIN SCREEN BACKGROUND (Sophia)
    main_screen = pygame.transform.scale(pygame.image.load("assets/ui/background.png").convert(), (SCREEN_WIDTH, SCREEN_HEIGHT))
    end_screen = pygame.transform.scale(pygame.image.load("assets/ui/theend.png").convert(), (SCREEN_WIDTH, SCREEN_HEIGHT))
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

    CUP_POS = (559, 545)
    CUP_SCALE = 1.0  
    served_base = None      
    served_toppings = []
    served_timer = 9999
    cup_shown = False
    cup_timer = 0            # for the pop-up animation (ms)

    # BASES (Sophia): click a jar on the right of the background to fill the cup
    BASE_ZONES = {
        "taro": pygame.Rect(1150, 320, 120, 110),      # purple jar
        "matcha": pygame.Rect(1035, 455, 130, 135),    # green jar
        "milktea": pygame.Rect(1170, 430, 105, 160),   # brown jar
    }
    selected_base = None     # None = empty cup

    # ICE (Sophia): customers ask for ice or no ice
    ice_image = pygame.image.load("assets/ice.png").convert_alpha()
    ice_added = False        # is there ice in the cup right now
    current_ice = False      # does the customer want ice
    served_ice = False
    ice_drop_start = 0
    ICE_LINES = ["With ice, please!", "Oh, and lots of ice!", "Ice please!"]
    NO_ICE_LINES = ["No ice, please!", "Oh, and no ice!", "Hold the ice!"]

    # CUPS + ICE BUTTONS (Sophia): click the cup stacks or the ice bucket (same as C / I)
    CUP_STACK_ZONE = pygame.Rect(920, 318, 150, 132)
    ICE_ZONE = pygame.Rect(725, 400, 158, 150)

    # TOPPING BUTTONS (Sophia): the 4 x 2 jars on the left of the background
    TOPPING_ZONES = {
        "pearls": pygame.Rect(75, 438, 85, 90),
        "coconutjelly": pygame.Rect(167, 438, 75, 90),
        "redbean": pygame.Rect(247, 438, 75, 90),
        "mangochunks": pygame.Rect(326, 438, 64, 90),
        "strawberrychunks": pygame.Rect(78, 560, 70, 90),
        "grassjelly": pygame.Rect(152, 560, 72, 90),
        "mangopearls": pygame.Rect(233, 560, 70, 90),
        "strawberrypearls": pygame.Rect(311, 560, 66, 90),
    }
    TOPPING_NAMES = {
        "pearls": "pearls", "coconutjelly": "coconut jelly", "redbean": "red bean",
        "mangochunks": "mango chunks", "strawberrychunks": "strawberry chunks",
        "grassjelly": "grass jelly", "mangopearls": "popping mango boba",
        "strawberrypearls": "popping strawberry boba",
    }
    topping_hover = {name: 0 for name in TOPPING_ZONES}   # hover animation, 0 to 1
    tag_font = pygame.font.Font(FONT_PATH, 24)

    # MONEY (Sophia)
    SAGE = (78, 104, 72)
    money_font = pygame.font.Font(FONT_PATH, 56)
    money_popup_text = ""
    displayed_money = 0      # the number on screen, slides toward money
    # TUTORIAL TEXT BOX
    tutorial_box = pygame.Rect(20, 110, 350, 250)
    tutorial_font = pygame.font.Font(FONT_PATH, 30)
    tutorial_hint_font = pygame.font.Font(FONT_PATH, 24)
    button_hover = {}        # hover animation for the custom buttons
    key_font = pygame.font.Font(FONT_PATH, 26)
    topping_drop_start = {}  # topping name -> time it was dropped in (for the bounce)
    # SCREEN TRANSITIONS (Sophia): crossfade whenever the screen changes
    last_screen = None
    transition_snapshot = None
    transition_timer = 0
    money_popup_timer = 0
    last_mood = "happy"

    # RECEIPT (Sophia)
    receipt_font = pygame.font.Font(FONT_PATH, 26)

    # KEY HINTS (Sophia): letter -> spot just above the thing on the counter
    KEY_HINTS = {
        "C": (1000, 315),    # cup stacks
        "I": (801, 382),     # ice bucket
    }

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
            "angry": ["Bro. Are u fr. I'm allergic to ts"],
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
    name_font = pygame.font.Font(FONT_PATH, 32)
    dialogue_box = pygame.Rect(16, 552, SCREEN_WIDTH - 32, 156)     # spans the whole bottom of the screen
    dialogue_close_button = pygame.Rect(dialogue_box.right - 58, dialogue_box.top + 12, 44, 36)
    dialogue_visible = True
    dialogue_timer = 0       # ms since the current line started (pop-in + typing)
    dialogue_fade = 0        # 0 = hidden, 1 = fully shown (smooth fade in/out)
    TYPE_SPEED = 0.045       # letters per ms (about 45 letters a second)
    CUSTOMER_DISPLAY_NAMES = {"hamster": "Hamster", "chiikawa": "Chiikawa", "peppapig": "Peppa",
                              "redpanda": "Red Panda", "snoopy": "Snoopy"}
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
                elif (event.key == K_RETURN or event.key == K_SPACE) and status == "game" and dialogue_text != "" and dialogue_visible:
                    # DIALOGUE: Enter finishes the text / closes it
                    advance_dialogue()
                elif event.key == K_1 and status == "mode_select":
                    status = "title"
                elif event.key == K_2 and status == "mode_select":
                    start_camera()
                    status = "title"
                elif event.key == K_s and status == "game" and not dialogue_visible and customer is not None and not customer_reacting:
                    # SERVE [S]
                    serve_drink()
                elif event.key == K_r and status == "game" and cup_shown and customer is not None and not customer_reacting:
                    # CLEAR [R]
                    clear_drink()
                elif event.key == K_c and status == "game":
                    # EMPTY CUP (Sophia)
                    get_cup()
                elif event.key == K_i and status == "game" and customer is not None and not customer_reacting:
                    toggle_ice()
            elif event.type == MOUSEBUTTONDOWN:
                if status == "mode_select":
                    if browser_button.collidepoint(event.pos):
                        status = "title"
                    elif download_button.collidepoint(event.pos):
                        start_camera()
                        status = "title"
                elif status in ("tutorial", "how_to_play"):
                    next_story_screen()
                elif status == "title":
                    if title_timer > 1400:
                        if start_rect.collidepoint(event.pos):
                            status = "tutorial"
                        elif quit_rect.collidepoint(event.pos):
                            running = False
                elif status == "game":
                    if dialogue_text != "" and dialogue_visible:
                        # DIALOGUE: X closes it, clicking anywhere else finishes the text / closes it
                        if dialogue_close_button.collidepoint(event.pos):
                            dialogue_visible = False
                        else:
                            advance_dialogue()
                    elif CUP_STACK_ZONE.collidepoint(event.pos):
                        get_cup()
                    elif customer is not None and not customer_reacting:
                        if serve_button.collidepoint(event.pos):
                            serve_drink()
                        elif clear_button.collidepoint(event.pos):
                            if cup_shown:
                                clear_drink()
                        elif ICE_ZONE.collidepoint(event.pos):
                            toggle_ice()
                        elif cup_shown:
                            # BASES: click a jar to fill the cup
                            for base_name, zone in BASE_ZONES.items():
                                if zone.collidepoint(event.pos):
                                    selected_base = base_name
                            # TOPPINGS: click a jar to add it (in order)
                            for topping_name, zone in TOPPING_ZONES.items():
                                if zone.collidepoint(event.pos):
                                    add_topping(topping_name)

        # SCREEN TRANSITIONS: the screen still shows last frame here, so keep a copy to fade out
        if (status, cutscene_index) != last_screen:
            if last_screen is not None:
                transition_snapshot = screen.copy()
                transition_timer = 0
            last_screen = (status, cutscene_index)

        screen.fill((0, 0, 0))

        if use_camera:
            ok, frame = camera.read()
            if ok:
                frame = cv2.flip(frame, 1)
                if status in ("game", "how_to_play"):
                    finger_count = count_all_fingers(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_image = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")

        if status == "mode_select":
            screen.blit(main_screen, (0, 0))
            dim = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
            dim.fill((40, 25, 15, 120))
            screen.blit(dim, (0, 0))
            pick_text = money_font.render("How are you playing?", True, (255, 244, 220))
            screen.blit(pick_text, pick_text.get_rect(center=(640, 220)))
            mouse_pos = pygame.mouse.get_pos()
            draw_ui_button("browser", browser_button, "Browser version", "[1]", (245, 228, 196), (110, 62, 30), mouse_pos)
            draw_ui_button("download", download_button, "Downloaded version", "[2]", (245, 228, 196), (110, 62, 30), mouse_pos)
        elif status == "title":
            # TITLE SCREEN
            title_timer += dt

            main_screen.set_alpha(min(255, int(255 * title_timer / 600)))
            screen.blit(main_screen, (0, 0))
            main_screen.set_alpha(255)

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
            pass
        elif status == "tutorial":
            screen.blit(cutscene[cutscene_index], (0, 0))
            if cutscene_index in STORY_LINES:
                # STORY (Sophia): visual-novel dialogue over the picture
                story_timer += dt
                story_frame_timer += dt
                text, shake = story_line()
                draw_story_box(text, shake)
            if cutscene_index in TUTORIAL_INSTRUCTIONS:
                draw_tutorial_text()
        elif status == "how_to_play":
            # TUTORIAL SPOT (Sophia): black screen for now, put the tutorial here later
            hint_text = font.render("click to start", True, (90, 90, 90))
            screen.blit(hint_text, hint_text.get_rect(center=(640, 680)))
            if use_camera:
                # HAND MODE TIPS (Sophia)
                if camera_image is not None:
                    draw_hand_preview(pygame.Rect(530, 24, 220, 165), False)
                # (text, colour, font): headings brown, the important rule in red
                BROWN, RED, DARK = (148, 90, 45), (170, 60, 50), (60, 40, 25)
                tip_lines = [
                    ("HAND MODE: your hand gestures are ONLY for the toppings!", RED, font),
                    ("- hold up 1-5 fingers to add a topping to the cup", DARK, name_font),
                    ("- each number is a secret topping: you have to GUESS which finger", DARK, name_font),
                    ("  count gives which topping (the box under the camera tells you)", DARK, name_font),
                    ("- EVERYTHING ELSE works like the tutorial: get a cup (C), pick the", DARK, name_font),
                    ("  base, add ice (I), then CLEAR (R) or SERVE (S) with the mouse/keys", DARK, name_font),
                    ("Before you start:", BROWN, font),
                    ("TEST: count 1-5 on your fingers, palm facing the screen!", RED, font),
                    ("- ensure you're in good lighting, and nothing is blocking your hand", DARK, name_font)

                ]
                # box grows to fit the text
                text_height = sum(line_font.get_linesize() + 4 for _, _, line_font in tip_lines) + 10
                tips = pygame.Rect(0, 0, 1060, text_height + 44)
                tips.midtop = (640, 205)
                pygame.draw.rect(screen, (255, 250, 238), tips, border_radius=16)
                pygame.draw.rect(screen, (148, 90, 45), tips, 3, border_radius=16)
                y = tips.top + 22
                for text, colour, line_font in tip_lines:
                    if line_font == font and y > tips.top + 30:
                        y += 10       # a little space before a heading
                    line_surface = line_font.render(text, True, colour)
                    screen.blit(line_surface, (tips.left + 36, y))
                    y += line_font.get_linesize() + 4
        elif status == "game":
            # MAIN SCREEN BACKGROUND
            screen.blit(main_screen, (0, 0))

            # CUSTOMER TIMER: next customer comes after the wait, served customers leave after 2 sec
            if customer is None and not customer_reacting:
                customer_timer += dt
                if customer_timer >= next_customer_time:
                    new_customer()
                    customer_timer = 0

            if customer_reacting:
                reaction_timer += dt
                if reaction_timer >= 2000:
                    customer = None
                    customer_reacting = False
                    reaction_timer = 0
                    customer_timer = 0
                    next_customer_time = random.randint(3000, 7000)

            # DRAW CUSTOMER
            if customer is not None:
                customer.draw(screen)

            # EMPTY CUP (Sophia): pops up with a small bounce, then shows the base and toppings
            if cup_shown:
                cup_timer += dt
                size = max(0.01, ease_out_back(cup_timer / 350)) * CUP_SCALE
                draw_cup(selected_base, selected_toppings, CUP_POS, size, 255, ice_added)

            # SERVED DRINK: slides up toward the customer and fades out
            served_timer += dt
            if served_timer < 450:
                t = served_timer / 450
                draw_cup(served_base, served_toppings, (CUP_POS[0], CUP_POS[1] - int(t * 120)), CUP_SCALE, int(255 * (1 - t)), served_ice)

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

            # CUPS + ICE (Sophia): clickable, same as C and I
            draw_counter_highlight(CUP_STACK_ZONE, "cups [C]", mouse_pos)
            draw_counter_highlight(ICE_ZONE, ("remove ice [I]" if ice_added else "ice [I]"), mouse_pos)

            # TOPPING BUTTONS (Sophia)
            draw_topping_buttons(mouse_pos)

            # KEY HINTS (Sophia)
            for letter, pos in KEY_HINTS.items():
                # hide the floating note while its thing is hovered (the hover tag shows the key instead)
                if letter == "C" and CUP_STACK_ZONE.collidepoint(mouse_pos):
                    continue
                if letter == "I" and ICE_ZONE.collidepoint(mouse_pos):
                    continue
                draw_key_hint(letter, pos)

            # SERVE + CLEAR BUTTONS
            draw_ui_button("serve", serve_button, "SERVE", "[S]", (122, 150, 112), (255, 248, 235), mouse_pos)
            draw_ui_button("clear", clear_button, "CLEAR", "[R]", (214, 128, 112), (255, 248, 235), mouse_pos)

            if use_camera:
                # OPENCV MODE (Sophia): toppings can also be picked with fingers
                if picking_toppings():
                    if finger_count == 0:
                        waiting_for_reset = False
                        hold_frames = 0
                    elif waiting_for_reset:
                        pass
                    elif finger_count > len(HAND_TOPPINGS) or HAND_TOPPINGS[finger_count - 1] not in available_toppings:
                        hold_frames = 0      # 9-10 fingers, or a locked topping: nothing to pick
                        held_count = finger_count
                    elif finger_count == held_count:
                        hold_frames += 1
                        if hold_frames >= HOLD_TIME and finger_count <= len(HAND_TOPPINGS):
                            add_topping_by_hand(HAND_TOPPINGS[finger_count - 1])
                            waiting_for_reset = True
                            hold_frames = 0
                    else:
                        held_count = finger_count
                        hold_frames = 0


            # HAND PREVIEW (Sophia): webcam with your tracked hands, the count, and the hold bar
            if use_camera and camera_image is not None:
                draw_hand_preview()

            # CUSTOMER DIALOGUE (Sophia)
            # new customer walks up: show their order line
            if customer is not None and customer is not dialogue_customer:
                dialogue_customer = customer
                reaction_given = False
                dialogue_visible = True
                dialogue_timer = 0
                dialogue_text = ""
                if customer.name in CUSTOMER_DIALOGUE:
                    dialogue_text = fill_in_dialogue(random.choice(CUSTOMER_DIALOGUE[customer.name]["order"]))
                    dialogue_text += " " + random.choice(ICE_LINES if current_ice else NO_ICE_LINES)

            # drink was served: switch to a reaction line (mood was worked out in serve_drink)
            if customer is not None and customer_reacting and not reaction_given:
                reaction_given = True
                dialogue_visible = True
                dialogue_timer = 0
                if customer.name in CUSTOMER_DIALOGUE:
                    dialogue_text = fill_in_dialogue(random.choice(CUSTOMER_DIALOGUE[customer.name][last_mood]))

            # customer left: the box fades out, then the text is cleared
            if customer is None:
                dialogue_visible = False
                if dialogue_fade < 0.02:
                    dialogue_text = ""

            # DRAW DIALOGUE (fades in and out smoothly)
            dialogue_timer += dt
            fade_target = 1 if (dialogue_visible and dialogue_text != "") else 0
            dialogue_fade += (fade_target - dialogue_fade) * 0.15
            if dialogue_text != "" and dialogue_fade > 0.02:
                draw_dialogue()

            # RECEIPT (Sophia): shows after you close the dialogue
            if customer is not None and not customer_reacting and not dialogue_visible:
                draw_receipt()

            # MONEY (Sophia)
            money_popup_timer += dt
            displayed_money += (money - displayed_money) * 0.08
            if abs(money - displayed_money) < 0.5:
                displayed_money = money
            draw_money()
        
        elif status == "end":
            # END SCREEN
            screen.blit(end_screen, (0, 0))

        # SCREEN TRANSITIONS: old screen fades out over the new one
        if transition_snapshot is not None:
            transition_timer += dt
            fade = 1 - transition_timer / 450
            if fade <= 0:
                transition_snapshot = None
            else:
                transition_snapshot.set_alpha(int(255 * fade))
                screen.blit(transition_snapshot, (0, 0))

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)

    if camera is not None:
        camera.release()
    pygame.quit()

if __name__ == "__main__":
    asyncio.run(main())