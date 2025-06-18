import os
import numpy as np
import random
import pretty_midi
from pretty_midi import PrettyMIDI, Instrument, Note
import moviepy.editor as mpy
import math
import pygame
from pydub import AudioSegment
import time
from tqdm import tqdm

# ============================== #
#          CONFIGURATION         #
# ============================== #

# === Audio / MIDI ===
SOUNDFONT = "soundfonts/FluidR3_GM.sf2"
AUDIO_NAME = "output.wav"
VIDEO_NAME = "tiktok.mp4"
TEMP_MIDI_PATH = "temp.mid"
MIDI_NOTE_GAIN_DB = 10
SFX_VOLUME_REDUCTION = 13
EXTRACT_MELODY = True
USE_SELECTED_INSTRUMENT = False
SELECTED_MIDI_INSTRUMENT = "piano"
MIDI_INSTRUMENTS = {
    "kalimba": 108, "marimba": 12, "woodblock": 115, "glass": 92,
    "musicbox": 10, "celesta": 8, "crystal": 99, "steel": 114,
    "bells": 14, "synth": 80, "piano": 0
}
MIDI_SONGS = {
    "blue": "songs/blue.mid",
    "mozart": "songs/mozart.mid",
    "countdown": "songs/countdown.mid",
    "mountain": "songs/mountain.mid",
    "mario": "songs/mario.mid",
    "unravel": "songs/unravel.mid",
    "tetris": "songs/tetris.mid",
    "nyan": "songs/nyan.mid",
    "birthday": "songs/birthday.mid",
    "frog": "songs/frog.mid",
    "dark": "songs/dark.mid",
    "fear": "songs/fear.mid"
}
SELECTED_MIDI = "blue"

# === Vidéo ===
FRAME_SIZE = (900, 1600)
FPS = 60
DURATION = 56
TEMP_FRAMES = "frames"

# === Couleurs utilitaires ===
GREEN = (100, 255, 100)
RED = (255, 100, 100)
BLUE = (100, 100, 255)
YELLOW = (255, 255, 100)
ORANGE = (255, 165, 0)
CYAN = (100, 255, 255)
MAGENTA = (255, 100, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# === Cercle ===
CIRCLE_FILL_MODE = "auto"
CIRCLE_COLORS = [WHITE]
CIRCLE_LINE_WIDTH = 8
BASE_RADIUS = 250
CIRCLE_SPACING = 30
TOTAL_CIRCLES = 1000
MAX_TOTAL_CIRCLES = 1000
CIRCLE_OFFSET_STEP = 7
MIN_CIRCLE_RADIUS = 220
CIRCLE_SHRINK_SPEED = 2
GAP_ENABLED = True
GAP_RATIO = 0.2
GAP_SIZE_DEGREES = 80
PHASE_OFFSET_DEGREES = 180
GAP_MODE = "aligned"
RECENTER_TRIGGER_RADIUS = 750
RECENTER_SHRINK_MULTIPLIER = 2.3
RESTART_RADIUS_AFTER_RECENTER = True
ROTATION_SPEED = 0.003

# === Balle ===
BALL_RADIUS = 50
BALL_OUTLINE_WIDTH = 6
BALL_FILL_COLOR = (0, 0, 0)
MAX_BALL_SPEED = 46
MIN_BALL_SPEED = 0
NOTE_DURATION = 0.4
GRAVITY = 0.3
ROTATION_INFLUENCE = 0.22
BOUNCINESS = 1.01
MAX_TRAIL = 13
BALL_COLORS = [
    GREEN,
    BLUE
]
SPAWN_MODE = "all"
SPAWN_INTERVAL_SECONDS = 5
BOUNCE_NOTE_COOLDOWN_FRAMES = 15
NOTE_ON_BALL_COLLISION = True
RANDOM_BOUNCE_ANGLE_DEGREES = 4
KEEP_BALL_COLOR = True
USE_CUSTOM_NAMES = True
BALL_NAMES = ["Yes", "No"]
USE_IMAGE_BALLS = False
BALL_IMAGE_PATHS = {
    0: "img/psg.png",
    1: "img/arsenal.png",
    2: "img/ball3.png",
    3: "img/ball4.png"
}

# === FX / Particules ===
# Son joué quand une balle casse un cercle
BALL_HIT_SOUNDS = {
        0: "yes.mp3",  # Balle 1
        1: "no.mp3",  # Balle 2
        2: "oof.wav",  # Balle 3
        3: "pew.wav",  # etc.
}
SFX_FOLDER = "hit_sfx"
SFX_COUNT = 6
PARTICLE_LIFESPAN = 30
PARTICLE_COUNT = 25
PARTICLE_SPEED = 4

# === Affichage / Textes ===
VIDEO_TITLE = "Is she the\none?"
SHOW_TITLE = True
NO_TEXTS = False
USE_TITLE_IMAGE = False
TITLE_IMAGE_PATH = "img/title.png"
TITLE_IMAGE_SCALE = 1.5
TITLE_FONT_SIZE = 70
SCORE_FONT_SIZE = 60
FINAL_SCORE_FONT_SIZE = 50
TITLE_Y_OFFSET = 200
TITLE_Y_OFFSET_IMG = 260
SCORE_Y_OFFSET = 250
FINAL_TITLE_Y = 150
FINAL_SCORE_Y_START = 300
FINAL_SCORE_LINE_SPACING = 80
FINAL_WINNER_Y = 800

# --- Variables globales ---
flash_alpha = 0
flash_color = (255, 255, 255)



# -----------------------------------------------------------------------------------------

NUM_BALLS = len(BALL_NAMES)
if NUM_BALLS > 2:
    TIMER_Y_OFFSET = 175
else:
    TIMER_Y_OFFSET = 100
FINAL_TITLE_Y = 150
FINAL_SCORE_Y_START = 300
FINAL_SCORE_LINE_SPACING = 80
FINAL_WINNER_Y = 800

TOTAL_FRAMES = FPS * DURATION


# --- INIT ---
os.makedirs(TEMP_FRAMES, exist_ok=True)
pygame.init()
pygame.display.set_mode((1, 1), pygame.HIDDEN)  # initialise un contexte vidéo invisible

title_image = None
if USE_TITLE_IMAGE and os.path.exists(TITLE_IMAGE_PATH):
    title_raw = pygame.image.load(TITLE_IMAGE_PATH).convert_alpha()
    w, h = title_raw.get_size()
    new_size = (int(w * TITLE_IMAGE_SCALE), int(h * TITLE_IMAGE_SCALE))
    title_image = pygame.transform.smoothscale(title_raw, new_size)
else:
    print("⚠️ Image du titre non trouvée ou désactivée.")


surface = pygame.Surface(FRAME_SIZE)
CENTER = np.array([450, 800])

# SFX setup
SFX_FOLDER = "hit_sfx"
SFX_COUNT = 6
broken_sfx_timestamps = []
particles = []

MIDI_PATH = MIDI_SONGS.get(SELECTED_MIDI)
if not MIDI_PATH or not os.path.exists(MIDI_PATH):
    raise FileNotFoundError(f"🎵 Fichier MIDI non trouvé : {MIDI_PATH}")

if EXTRACT_MELODY:
    original_midi = pretty_midi.PrettyMIDI(MIDI_PATH)

    melodic_tracks = [
        inst for inst in original_midi.instruments
        if not inst.is_drum and len(inst.notes) > 0
    ]

    if not melodic_tracks:
        raise ValueError("⚠️ Aucune piste mélodique trouvée.")

    melody_instr = max(
        melodic_tracks,
        key=lambda inst: sum(n.end - n.start for n in inst.notes)
    )

    track_name = melody_instr.name if melody_instr.name else "(sans nom)"
    print(f"🎼 Piste mélodique sélectionnée : {track_name} ({len(melody_instr.notes)} notes)")

    melody_only = pretty_midi.PrettyMIDI()
    melody_only.instruments.append(melody_instr)
    melody_only.write(TEMP_MIDI_PATH)

    MIDI_PATH = TEMP_MIDI_PATH

# Chargement final
midi_data = pretty_midi.PrettyMIDI(MIDI_PATH)

all_notes = [n for inst in midi_data.instruments for n in inst.notes]
all_notes.sort(key=lambda n: n.start)
note_index = 0


# Police pour le timer
font_timer = pygame.font.SysFont("Arial", 48, bold=True)

# Positionnement (par exemple, centré en bas à 85% de la hauteur)
timer_y = int(FRAME_SIZE[1] * 0.85)
timer_x = FRAME_SIZE[0] // 2

def load_circular_image(image_path, radius):
    raw = pygame.image.load(image_path).convert_alpha()
    img = pygame.transform.smoothscale(raw, (2 * radius, 2 * radius))

    # Créer une surface ronde avec alpha
    circle_surf = pygame.Surface((2 * radius, 2 * radius), pygame.SRCALPHA)

    # Masque circulaire
    pygame.draw.circle(circle_surf, (255, 255, 255, 255), (radius, radius), radius)
    img.blit(circle_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    return img


def draw_timer_bar(surface, progress, pos, size, border_radius=14):
    """
    Draws a gradient progress bar.
    :param surface: The surface to draw on
    :param progress: 0.0 to 1.0 (1.0 = full bar, 0 = empty)
    :param pos: (x, y) top-left position
    :param size: (width, height) size of the bar
    """
    x, y = pos
    width, height = size

    # Fond glassy
    glass = pygame.Surface((width, height), pygame.SRCALPHA)
    draw_glass_rect(glass, glass.get_rect(), alpha=100, border_radius=border_radius)
    surface.blit(glass, (x, y))

    # Barre de progression avec gradient
    bar_width = int(width * progress)
    if bar_width > 0:
        gradient = pygame.Surface((bar_width, height), pygame.SRCALPHA)
        for i in range(bar_width):
            r = int(255 * (1 - i / width))  # du rouge au vert
            g = int(255 * (i / width))
            pygame.draw.line(gradient, (r, g, 80, 230), (i, 0), (i, height))
        surface.blit(gradient, (x, y))


def get_multiline_height(text, font, line_spacing=5):
    lines = text.split("\n")
    line_heights = [font.render(line, True, (0,0,0)).get_height() for line in lines]
    return sum(line_heights) + (len(lines) - 1) * line_spacing

def darken_color(color, factor=0.6):
    return tuple(int(c * factor) for c in color)

# Dessin avec fond blanc arrondi (tu peux custom)
def draw_timer(surface, text, pos, font):
    text_surf = font.render(text, True, (0, 0, 0))
    padding_x, padding_y = 28, 14
    rect = text_surf.get_rect(center=pos)
    bg_rect = pygame.Rect(
        rect.left - padding_x, rect.top - padding_y,
        rect.width + 2 * padding_x, rect.height + 2 * padding_y
    )
    # Fond blanc semi-transparent
    timer_bg = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
    timer_bg.fill((255, 255, 255, 210))
    pygame.draw.rect(timer_bg, (255, 255, 255, 210), timer_bg.get_rect(), border_radius=28)
    surface.blit(timer_bg, (bg_rect.left, bg_rect.top))
    surface.blit(text_surf, rect)



def draw_text_centered_on_ball(surface, text, position, max_radius, color):
    # Taille de police estimée par rapport au rayon
    font_size = max(10, int(max_radius * 0.6))  # Ajustable
    font = pygame.font.SysFont("Arial", font_size, bold=True)

    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=position)

    # Si le texte est trop large, on réduit la taille
    while (text_rect.width > max_radius * 1.8) and font_size > 10:
        font_size -= 1
        font = pygame.font.SysFont("Arial", font_size, bold=True)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect(center=position)

    surface.blit(text_surface, text_rect)


def draw_caption_text(surface, text, pos, font, padding_x=20, padding_y=10,
                      text_color=(0, 0, 0), bg_color=(255, 255, 255), center=True, border_radius=12, line_spacing=5):
    lines = text.split("\n")
    text_surfaces = [font.render(line, True, text_color) for line in lines]

    max_width = max(surf.get_width() for surf in text_surfaces)
    total_height = sum(surf.get_height() for surf in text_surfaces) + (len(lines) - 1) * line_spacing

    # Calcul du rectangle de fond
    rect = pygame.Rect(0, 0, max_width + 2 * padding_x, total_height + 2 * padding_y)

    if center:
        rect.center = pos
    else:
        rect.topleft = pos

    # Capsule blanche
    caption_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(caption_surf, bg_color, caption_surf.get_rect(), border_radius=border_radius)

    # Blit du fond
    surface.blit(caption_surf, (rect.left, rect.top))

    # Blit des lignes de texte
    y = rect.top + padding_y
    for line_surf in text_surfaces:
        line_rect = line_surf.get_rect(centerx=rect.centerx, top=y)
        surface.blit(line_surf, line_rect)
        y += line_surf.get_height() + line_spacing



def draw_scores(surface, balls, font, y_offset=250):
    active_balls = [b for b in balls if b.active]
    count = len(active_balls)

    if count <= 2:
        # Cas simple : tous sur une ligne
        texts = []
        total_width = 0
        padding = 40
        for ball in active_balls:
            text = f"{ball.name} : {ball.score}"
            surf = font.render(text, True, ball.color)
            bubble_width = surf.get_width() + 2 * 20
            texts.append((text, bubble_width, ball.color))
            total_width += bubble_width + padding
        total_width -= padding
        x = CENTER[0] - total_width // 2
        for text, width, color in texts:
            dark_bg = darken_color(color, 0.55)
            draw_caption_text(surface, text, (x + width // 2, y_offset), font, text_color=(255, 255, 255), bg_color=dark_bg)
            x += width + padding

    else:
        # Cas 2 lignes (moitié haute, moitié basse)
        mid = (count + 1) // 2
        row1 = active_balls[:mid]
        row2 = active_balls[mid:]

        def draw_row(row, y):
            texts = []
            total_width = 0
            padding = 40
            for ball in row:
                text = f"{ball.name} : {ball.score}"
                surf = font.render(text, True, ball.color)
                bubble_width = surf.get_width() + 2 * 20
                texts.append((text, bubble_width, ball.color))
                total_width += bubble_width + padding
            total_width -= padding
            x = CENTER[0] - total_width // 2
            for text, width, color in texts:
                draw_caption_text(surface, text, (x + width // 2, y), font, text_color=color)
                x += width + padding

        draw_row(row1, y_offset)
        draw_row(row2, y_offset + 90)  # ou ajuste ce spacing vertical





def draw_text_with_outline(surface, text, pos, font, text_color, outline_color=(0, 0, 0), outline_width=2, center=False):
    text_surface = font.render(text, True, text_color)
    if center:
        pos = (pos[0] - text_surface.get_width() // 2, pos[1])
    x, y = pos
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx == 0 and dy == 0:
                continue
            outline_surface = font.render(text, True, outline_color)
            surface.blit(outline_surface, (x + dx, y + dy))
    surface.blit(text_surface, (x, y))


# --- RotatingCircle ---
class RotatingCircle:
    def __init__(self, radius, gap_enabled=True, gap_angle_deg=45, phase_offset=0, color=(255, 255, 255), line_width=2):
        self.initial_radius = radius
        self.radius = radius
        self.active = True
        self.gap_enabled = gap_enabled
        self.gap_start_deg = phase_offset
        self.gap_size_deg = gap_angle_deg
        self.color = color
        self.line_width = line_width

    def angle_to_ball(self, ball_pos):
        vec = ball_pos - CENTER
        angle = math.degrees(math.atan2(-vec[1], vec[0])) % 360
        return angle


    def is_in_gap(self, angle, rotation_angle):
        if(GAP_ENABLED == False):
            return False
        angle = angle % 360
        gap_start = (self.gap_start_deg + math.degrees(rotation_angle)) % 360
        gap_end = (gap_start + self.gap_size_deg) % 360

        if gap_start < gap_end:
            return gap_start <= angle <= gap_end
        else:
            return angle >= gap_start or angle <= gap_end




    def update_gap(self):
        self.gap_start_deg = (self.gap_start_deg + math.degrees(ROTATION_SPEED)) % 360

    def update_radius(self, shrink_multiplier=1.0):
        if not self.active:
            return
        if self.radius > MIN_CIRCLE_RADIUS:
            self.radius -= CIRCLE_SHRINK_SPEED * shrink_multiplier
            self.radius = max(self.radius, MIN_CIRCLE_RADIUS)


    def draw(self, surface, rotation_angle):
        rect = pygame.Rect(
            CENTER[0] - self.radius,
            CENTER[1] - self.radius,
            2 * self.radius,
            2 * self.radius
        )
        start_angle = math.radians(self.gap_start_deg + rotation_angle * 180 / math.pi)
        end_angle = math.radians(self.gap_start_deg + self.gap_size_deg + rotation_angle * 180 / math.pi)
        
        # Dessine tout le cercle sauf l'arc du gap
        pygame.draw.arc(surface, self.color, rect, end_angle, start_angle + 2 * math.pi, self.line_width)




# --- Particle Class ---
class Particle:
    def __init__(self, radius, color):
        angle = random.uniform(0, 2 * math.pi)
        self.position = CENTER + np.array([math.cos(angle), math.sin(angle)]) * radius
        speed = random.uniform(0.5 * PARTICLE_SPEED, PARTICLE_SPEED)
        direction = np.array([math.cos(angle), math.sin(angle)])
        self.velocity = direction * speed
        self.life = PARTICLE_LIFESPAN
        self.color = color

    def update(self):
        self.position += self.velocity
        self.life -= 1

    def draw(self, surface):
        if self.life > 0:
            alpha = int(255 * (self.life / PARTICLE_LIFESPAN))
            r, g, b = self.color
            color = (r, g, b, alpha)
            particle_surf = pygame.Surface(FRAME_SIZE, pygame.SRCALPHA)
            pygame.draw.circle(particle_surf, color, self.position.astype(int), 3)
            surface.blit(particle_surf, (0, 0))

# --- Ball Class ---
class Ball:

    def __init__(self, spawn_frame, i):
        self.spawn_frame = spawn_frame
        self.active = False
        self.pos = CENTER.astype(float)
        self.vel = np.array([random.uniform(-5, 5), -6.0])
        self.trail = []
        self.color = BALL_COLORS[i % len(BALL_COLORS)]

        self.color_index = i % len(BALL_COLORS)
        self.score = 0
        self.index = i  # pour savoir si c'est Balle 1, 2, etc.
        self.name = BALL_NAMES[i] if USE_CUSTOM_NAMES else f"Balle {i+1}"
        self.last_bounce_frame = -999  # Pour éviter les notes spam
        self.bounce_cooldown_frames = BOUNCE_NOTE_COOLDOWN_FRAMES
        if USE_IMAGE_BALLS and i < len(BALL_IMAGE_PATHS):
            image_path = BALL_IMAGE_PATHS[i]
            self.image = load_circular_image(image_path, BALL_RADIUS)
            self.image_small = load_circular_image(image_path, 32)  # pour le score
        else:
            self.image = None
            self.image_small = None


# --- Utility Functions ---
def reflect_velocity_with_rotation(pos, vel):
    normal = pos - CENTER
    normal /= np.linalg.norm(normal)
    reflected = vel - 2 * np.dot(vel, normal) * normal

    # Influence de la rotation
    tangent = np.array([-normal[1], normal[0]])
    tangent_speed = ROTATION_SPEED * np.linalg.norm(normal)
    final = reflected * BOUNCINESS + tangent * (tangent_speed * ROTATION_INFLUENCE)

    # Ajout de bruit aléatoire (léger)
    if RANDOM_BOUNCE_ANGLE_DEGREES > 0:
        angle_offset = math.radians(random.uniform(-RANDOM_BOUNCE_ANGLE_DEGREES, RANDOM_BOUNCE_ANGLE_DEGREES))
        current_angle = math.atan2(final[1], final[0])
        new_angle = current_angle + angle_offset
        speed = np.linalg.norm(final)
        final = np.array([math.cos(new_angle), math.sin(new_angle)]) * speed

    return final


def balls_collide(ball1, ball2):
    distance = np.linalg.norm(ball1.pos - ball2.pos)
    return distance <= 2 * BALL_RADIUS

def draw_glass_rect(surface, rect, alpha=160, border_radius=18, border_alpha=200):
    glass = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    
    # Fond semi-transparent blanc
    glass.fill((255, 255, 255, alpha))
    
    # Bordure claire
    pygame.draw.rect(glass, (255, 255, 255, border_alpha), glass.get_rect(), width=2, border_radius=border_radius)
    
    # Applique sur le canvas
    surface.blit(glass, rect.topleft)


# --- Init Circles ---
circles = []
gap_angle = GAP_SIZE_DEGREES if GAP_ENABLED else 0
random_offset = random.randint(0, 359)

for i in range(TOTAL_CIRCLES):
    radius = BASE_RADIUS + i * CIRCLE_SPACING
    base_offset = random_offset if GAP_MODE == "aligned" else (random_offset + (PHASE_OFFSET_DEGREES if i % 2 == 1 else 0)) % 360
    offset = (base_offset + i * CIRCLE_OFFSET_STEP) % 360
    color = CIRCLE_COLORS[i % len(CIRCLE_COLORS)]
    circles.append(RotatingCircle(radius, GAP_ENABLED, gap_angle, phase_offset=offset, color=color, line_width=CIRCLE_LINE_WIDTH))


# --- Init Balls ---
balls = []
for i in range(NUM_BALLS):
    if SPAWN_MODE == "all":
        spawn = 0
    elif SPAWN_MODE == "interval":
        spawn = int(i * SPAWN_INTERVAL_SECONDS * FPS)
    else:
        raise ValueError("SPAWN_MODE must be 'all' or 'interval'")
    balls.append(Ball(spawn_frame=spawn, i=i))


played_notes = []
rotation_angle = 0
broken_circle_count = 0

# --- MAIN LOOP ---
for frame in tqdm(range(TOTAL_FRAMES), desc="🎬 Génération frames"):
    frame_start = time.time()

    t = frame / FPS
    surface.fill((10, 10, 20))

    font_big = pygame.font.SysFont("Arial", TITLE_FONT_SIZE, bold=True)
    font_small = pygame.font.SysFont("Arial", SCORE_FONT_SIZE, bold=True)



    # --- Gestion fixe des cercles --- #
    rotation_angle += ROTATION_SPEED

    # Trigger du recentrage
    max_radius = max(c.radius for c in circles if c.active)
    recenter_mode = max_radius > RECENTER_TRIGGER_RADIUS

    # Recentrage forcé si nécessaire
    min_radius = min([c.radius for c in circles if c.active], default=BASE_RADIUS)
    for circle in circles:
        circle.update_gap()
        shrink_speed = RECENTER_SHRINK_MULTIPLIER if recenter_mode else 1.0

        if min_radius > MIN_CIRCLE_RADIUS or circle.radius <= min_radius:
            circle.update_radius(shrink_multiplier=shrink_speed)

        if not circle.active:
            continue
        if circle.radius < BASE_RADIUS - 100 or circle.radius > FRAME_SIZE[1] + 200:
            continue  # hors champ

        if circle.active:
            circle.draw(surface, rotation_angle)

    # Reset complet si tout est shrinké
    if recenter_mode and all(c.radius <= MIN_CIRCLE_RADIUS for c in circles if c.active):
        for i, circle in enumerate(circles):
            circle.radius = BASE_RADIUS + i * CIRCLE_SPACING
            circle.active = True


    # Nettoyage : supprime les cercles trop petits et inactifs
    circles = [c for c in circles if c.radius > MIN_CIRCLE_RADIUS or c.active]

    # Réinitialisation après recentrage total
    if recenter_mode and all(c.radius <= MIN_CIRCLE_RADIUS for c in circles):
        if RESTART_RADIUS_AFTER_RECENTER:
            next_circle_radius = BASE_RADIUS
            circles.clear()  # Vide la liste pour recommencer frais

    active_balls = [b for b in balls if frame >= b.spawn_frame]
    for ball in active_balls:
        if not ball.active:
            ball.active = True

        ball.vel[1] += GRAVITY
        speed = np.linalg.norm(ball.vel)

        # Limite supérieure
        if speed > MAX_BALL_SPEED:
            ball.vel = (ball.vel / speed) * MAX_BALL_SPEED

        # Limite inférieure
        elif speed < MIN_BALL_SPEED:
            # On boost dans la direction actuelle
            if speed > 0:
                ball.vel = (ball.vel / speed) * MIN_BALL_SPEED
            else:
                # Si à l'arrêt, on donne une direction aléatoire
                angle = random.uniform(0, 2 * math.pi)
                ball.vel = np.array([math.cos(angle), math.sin(angle)]) * MIN_BALL_SPEED

        ball.pos += ball.vel

        ball.trail.append(ball.pos.copy())
        if len(ball.trail) > MAX_TRAIL:
            ball.trail.pop(0)

        rel = ball.pos - CENTER
        dist = np.linalg.norm(rel)
        for circle in circles:
            if not circle.active:
                continue
            if abs(dist - circle.radius) <= BALL_RADIUS:
                angle = circle.angle_to_ball(ball.pos)
                if circle.is_in_gap(angle, rotation_angle):
                    circle.active = False
                    # On utilise le son spécifique à la balle si défini, sinon fallback
                    sound_filename = BALL_HIT_SOUNDS.get(ball.index, f"wi{broken_circle_count % SFX_COUNT + 1}.wav")
                    broken_sfx_timestamps.append((t, sound_filename))

                    broken_circle_count += 1
                    ball.score += 1
                    flash_alpha = 80
                    flash_color = ball.color

                    for _ in range(PARTICLE_COUNT):
                        particles.append(Particle(circle.radius, ball.color))
                    break

                direction = rel / dist
                ball.pos = CENTER + direction * (circle.radius - BALL_RADIUS + 0.5)  # petit buffer

                ball.vel = reflect_velocity_with_rotation(ball.pos, ball.vel)

                if frame - ball.last_bounce_frame >= ball.bounce_cooldown_frames:
                    note = all_notes[note_index]
                    played_notes.append((t, note.pitch, note.velocity))
                    note_index = (note_index + 1) % len(all_notes)
                    ball.last_bounce_frame = frame

                if not KEEP_BALL_COLOR:
                    ball.color_index = (ball.color_index + 1) % len(BALL_COLORS)
                    ball.color = BALL_COLORS[ball.color_index]
                break

        for i, p in enumerate(ball.trail):
            alpha = int(60 * (i + 1) / MAX_TRAIL)
            color = (*ball.color, alpha)
            trail_surf = pygame.Surface(FRAME_SIZE, pygame.SRCALPHA)
            pygame.draw.circle(trail_surf, color, p.astype(int), BALL_RADIUS)
            surface.blit(trail_surf, (0, 0))

        # Outline
        pygame.draw.circle(surface, ball.color, ball.pos.astype(int), BALL_RADIUS)

        if ball.image:
            img_pos = ball.pos - np.array([BALL_RADIUS, BALL_RADIUS])
            surface.blit(ball.image, img_pos.astype(int))
        else:
            pygame.draw.circle(surface, BALL_FILL_COLOR, ball.pos.astype(int), BALL_RADIUS - BALL_OUTLINE_WIDTH)

        if(USE_CUSTOM_NAMES):
            draw_text_centered_on_ball(surface, ball.name, ball.pos.astype(int), BALL_RADIUS, (255, 255, 255))

    for i in range(len(active_balls)):
        for j in range(i + 1, len(active_balls)):
            b1, b2 = active_balls[i], active_balls[j]
            if balls_collide(b1, b2):
                direction = b1.pos - b2.pos
                distance = np.linalg.norm(direction)
                
                if distance == 0:
                    # Cas très rare où les deux balles sont exactement au même endroit
                    direction = np.random.randn(2)
                    direction /= np.linalg.norm(direction)
                    distance = 0.01
                else:
                    direction /= distance

                # Repositionnement : écarte les deux balles pour éviter l’overlap
                overlap = 2 * BALL_RADIUS - distance
                correction = direction * (overlap / 2)
                b1.pos += correction
                b2.pos -= correction

                # Inversion simple des vitesses projetées sur la direction de collision
                v1_proj = np.dot(b1.vel, direction)
                v2_proj = np.dot(b2.vel, direction)
                
                b1.vel += (v2_proj - v1_proj) * direction
                b2.vel += (v1_proj - v2_proj) * direction
                if NOTE_ON_BALL_COLLISION:
                    if frame - b1.last_bounce_frame >= b1.bounce_cooldown_frames:
                        note = all_notes[note_index]
                        played_notes.append((t, note.pitch, note.velocity))
                        note_index = (note_index + 1) % len(all_notes)
                        b1.last_bounce_frame = frame
                        b2.last_bounce_frame = frame  # on met à jour les deux

    for particle in particles[:]:
        particle.update()
        particle.draw(surface)
        if particle.life <= 0:
            particles.remove(particle)

    if flash_alpha > 0:
        overlay = pygame.Surface(FRAME_SIZE, pygame.SRCALPHA)
        overlay.fill((*flash_color, flash_alpha))
        surface.blit(overlay, (0, 0))
        flash_alpha = max(0, flash_alpha - 5)

    # Texte toujours au premier plan
    # Calcul dynamique de la hauteur du titre
    title_height = get_multiline_height(VIDEO_TITLE, font_big)
    score_y_offset = TITLE_Y_OFFSET + title_height + 25 # petit espace en plus

    # Calcul du temps restant
    time_left = max(0, DURATION - (frame // FPS))
    minutes = time_left // 60
    seconds = time_left % 60
    timer_str = f"Time left: {seconds:02d}"

    # Affichage du timer
    timer_font = pygame.font.SysFont("Arial", 38, bold=True)
    text_surf = timer_font.render(timer_str, True, (0, 0, 0))

    padding_x, padding_y = 24, 12
    bg_rect = pygame.Rect(
        0, 0,
        text_surf.get_width() + 2 * padding_x,
        text_surf.get_height() + 2 * padding_y
    )
    bg_rect.center = (FRAME_SIZE[0] // 2, score_y_offset + TIMER_Y_OFFSET)  # au-dessus des scores

    # Fond noir semi-transparent avec arrondi
    if not NO_TEXTS:
        timer_box = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(timer_box, (255, 255, 255, 250), timer_box.get_rect(), border_radius=20)
        timer_box.blit(text_surf, (padding_x, padding_y))

        # Blit sur l’écran
        surface.blit(timer_box, bg_rect.topleft)




    # Hauteur dynamique du titre (si affiché)
    if USE_TITLE_IMAGE and title_image:
        rect = title_image.get_rect(center=(CENTER[0], TITLE_Y_OFFSET_IMG))
        surface.blit(title_image, rect)
    else:
        if SHOW_TITLE:
            draw_caption_text(surface, VIDEO_TITLE, (CENTER[0], TITLE_Y_OFFSET), font_big)

    if not NO_TEXTS:
        draw_scores(surface, balls, font_small, y_offset=score_y_offset)




    pygame.image.save(surface, f"{TEMP_FRAMES}/frame_{frame:04d}.png")
    
    rotation_angle += ROTATION_SPEED
    
    #print(f"Frame {frame} took {time.time() - frame_start:.3f} seconds")

FINAL_DURATION = 5
FINAL_FRAMES = int(FPS * FINAL_DURATION)

final_text_font = pygame.font.SysFont("Arial", TITLE_FONT_SIZE, bold=True)
final_score_font = pygame.font.SysFont("Arial", FINAL_SCORE_FONT_SIZE, bold=True)

scores = [b.score for b in balls]
max_score = max(scores)
winners = [i for i, s in enumerate(scores) if s == max_score]
winner_text = "Draw! We don't know." if len(winners) > 1 else f"{balls[winners[0]].name} is the answer!"

for i in range(FINAL_FRAMES):
    surface.fill((10, 10, 20))

    draw_text_with_outline(surface, "Final Scores", (CENTER[0], FINAL_TITLE_Y), final_text_font, (255, 255, 255), center=True)
    
    for idx, ball in enumerate(balls):
        score_text = f"{ball.name} : {ball.score}"
        y = FINAL_SCORE_Y_START + idx * FINAL_SCORE_LINE_SPACING
        dark_bg = darken_color(color, 0.55)
        draw_caption_text(surface, score_text, (CENTER[0], y), final_score_font, text_color=ball.color, bg_color=dark_bg)


    winner_color = (255, 255, 255) if len(winners) > 1 else balls[winners[0]].color
    draw_caption_text(surface, winner_text, (CENTER[0], FINAL_WINNER_Y), final_text_font, text_color=winner_color, bg_color=(255, 255, 255))

    pygame.image.save(surface, f"{TEMP_FRAMES}/frame_{TOTAL_FRAMES + i:04d}.png")


pygame.quit()

# --- MIDI OUTPUT ---
midi_out = PrettyMIDI()
if not USE_SELECTED_INSTRUMENT and len(midi_data.instruments) == 1:
    instr = midi_data.instruments[0]
    instrument_id = instr.program
    print(f"🎹 Instrument détecté dans le fichier MIDI : {pretty_midi.program_to_instrument_name(instrument_id)} (program={instrument_id})")
else:
    instrument_id = MIDI_INSTRUMENTS.get(SELECTED_MIDI_INSTRUMENT, 0)
    print(f"🎹 Instrument sélectionné manuellement : {pretty_midi.program_to_instrument_name(instrument_id)} (program={instrument_id})")

piano = Instrument(program=instrument_id)

for t, pitch, vel in played_notes:
    note = Note(velocity=vel, pitch=pitch, start=t, end=t + NOTE_DURATION)
    piano.notes.append(note)
midi_out.instruments.append(piano)
midi_out.write("rebound.mid")

# --- AUDIO OUTPUT ---
os.system(f"fluidsynth -ni {SOUNDFONT} rebound.mid -F {AUDIO_NAME} -r 44100")
main_audio = AudioSegment.from_file(AUDIO_NAME)
main_audio += MIDI_NOTE_GAIN_DB
sfx_mix = AudioSegment.silent(duration=len(main_audio))

for t, sfx_name in broken_sfx_timestamps:
    sfx_path = os.path.join(SFX_FOLDER, sfx_name)
    if os.path.exists(sfx_path):
        sfx = AudioSegment.from_file(sfx_path) - SFX_VOLUME_REDUCTION
        position_ms = int(t * 1000)
        sfx_mix = sfx_mix.overlay(sfx, position=position_ms)
    else:
        print(f"⚠️ Fichier SFX manquant : {sfx_path}")


final_audio = main_audio.overlay(sfx_mix)
final_audio.export(AUDIO_NAME, format="wav")

# --- VIDEO OUTPUT ---
image_files = sorted([os.path.join(TEMP_FRAMES, f) for f in os.listdir(TEMP_FRAMES) if f.endswith(".png")])
clip = mpy.ImageSequenceClip(image_files, fps=FPS)
audio = mpy.AudioFileClip(AUDIO_NAME)
clip = clip.set_audio(audio)
clip.write_videofile(VIDEO_NAME, codec='libx264', audio_codec='aac')

print(f"✅ Vidéo générée avec cercles optimisés et gaps plus rapides : {VIDEO_NAME}")