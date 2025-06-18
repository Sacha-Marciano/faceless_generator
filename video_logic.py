import pygame
import moviepy.editor as mpy
import os
import random
import math
import os
from moviepy.config import change_settings

# Set the ImageMagick binary path
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

def generate_video(caption: str, text1: str, text2: str, duration: int = 10) -> str:
    WIDTH, HEIGHT = 800, 600
    FPS = 30
    BALL_RADIUS = 10
    CIRCLE_RADIUS = 60
    NUM_CIRCLES = 10
    CIRCLE_GAP_ANGLE = 40

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 30)

    frames = []

    class Ball:
        def __init__(self, x, y, vx, vy, text):
            self.x = x
            self.y = y
            self.vx = vx
            self.vy = vy
            self.text = text
            self.score = 0

        def update(self):
            self.vy += 0.5
            self.x += self.vx
            self.y += self.vy
            if self.y + BALL_RADIUS > HEIGHT:
                self.y = HEIGHT - BALL_RADIUS
                self.vy *= -0.8
            if self.x - BALL_RADIUS < 0 or self.x + BALL_RADIUS > WIDTH:
                self.vx *= -1

        def draw(self):
            pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), BALL_RADIUS)
            label = font.render(self.text, True, (255, 255, 255))
            screen.blit(label, (int(self.x + 10), int(self.y - 10)))

    class Ring:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.broken = False

        def draw(self):
            if self.broken:
                return
            gap_rad = math.radians(CIRCLE_GAP_ANGLE)
            pygame.draw.circle(screen, (255, 255, 255), (self.x, self.y), CIRCLE_RADIUS, 2)
            pygame.draw.arc(screen, (0, 0, 0),
                            (self.x - CIRCLE_RADIUS, self.y - CIRCLE_RADIUS, CIRCLE_RADIUS*2, CIRCLE_RADIUS*2),
                            -gap_rad/2, gap_rad/2, CIRCLE_RADIUS)

        def check_collision(self, ball):
            if self.broken:
                return
            dx = ball.x - self.x
            dy = ball.y - self.y
            dist = math.hypot(dx, dy)
            if dist < CIRCLE_RADIUS + BALL_RADIUS:
                angle = math.atan2(dy, dx)
                if abs(angle) < math.radians(CIRCLE_GAP_ANGLE / 2):
                    self.broken = True
                    ball.score += 1
                else:
                    ball.vx *= -1
                    ball.vy *= -1

    balls = [
        Ball(100, 100, 3, 0, text1),
        Ball(200, 100, -3, 0, text2)
    ]
    rings = [Ring(random.randint(100, WIDTH-100), random.randint(100, HEIGHT-200)) for _ in range(NUM_CIRCLES)]

    frame_count = duration * FPS
    for _ in range(frame_count):
        screen.fill((0, 0, 0))
        for ring in rings:
            ring.draw()
        for ball in balls:
            ball.update()
            for ring in rings:
                ring.check_collision(ball)
            ball.draw()

        score_text = f"{text1}: {balls[0].score}  |  {text2}: {balls[1].score}"
        score_render = font.render(score_text, True, (255, 255, 255))
        screen.blit(score_render, (WIDTH // 2 - score_render.get_width() // 2, 10))

        frame_surface = screen.copy()
        frames.append(pygame.surfarray.array3d(frame_surface).swapaxes(0, 1))
        clock.tick(FPS)
        pygame.display.flip()

    pygame.quit()

    filename = "output.mp4"
    video = mpy.ImageSequenceClip(frames, fps=FPS)
    if caption:
        txt_clip = mpy.TextClip(caption, fontsize=40, color='white').set_duration(3).set_position('center')
        video = mpy.concatenate_videoclips([txt_clip.set_duration(2), video])
    video.write_videofile(filename, fps=FPS)
    return filename
