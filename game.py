import pygame
import os
import random

BIRD_IMG = [
    pygame.image.load(os.path.join("imgs", "bird1.png")),
    pygame.image.load(os.path.join("imgs", "bird2.png")),
    pygame.image.load(os.path.join("imgs", "bird3.png")),
]

BG_IMG = pygame.image.load(os.path.join("imgs", "bg.png"))
BG_IMG = pygame.transform.scale(BG_IMG, (BG_IMG.get_width() * 2, BG_IMG.get_height()))

SCREEN_WIDTH = BG_IMG.get_width()
SCREEN_HEIGHT = BG_IMG.get_height()

win = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

bg_x = (SCREEN_WIDTH - BG_IMG.get_width()) // 2
bg_y = (SCREEN_HEIGHT - BG_IMG.get_height()) // 2


SCREEN_WIDTH = BG_IMG.get_width()
SCREEN_HEIGHT = BG_IMG.get_height()

BASE_IMG = pygame.image.load(os.path.join("imgs", "base.png"))
BASE_IMG = pygame.transform.scale(BASE_IMG, (SCREEN_WIDTH, BASE_IMG.get_height()))

base_width = BASE_IMG.get_width()
base_height = BASE_IMG.get_height()
base_x = 0
base_y = SCREEN_HEIGHT - base_height

clock = pygame.time.Clock()


class BIRD:
    GRAVITY = 0.5
    JUMP_STRENGTH = -10
    ROTATION_SPEED = 10

    def __init__(self, x=50, y=200):
        self.x = x
        self.y = y
        self.rot = 0
        self.vel = 0
        self.current_img_tick = 0

    def move(self):
        self.vel += self.GRAVITY
        self.y += self.vel

        if self.vel < 0:
            self.rot = max(
                25, self.rot - self.ROTATION_SPEED
            )  # Rotate upwards (limit to -25°)
        else:  # If falling down
            self.rot = min(
                -90, self.rot + self.ROTATION_SPEED
            )  # Rotate downwards (limit to 90°)

    def jump(self):
        """Make the bird jump"""
        self.vel = self.JUMP_STRENGTH
        self.rot = -25

    def draw(self, win):
        """Draw the bird with rotation"""
        rotated_image = pygame.transform.rotate(
            BIRD_IMG[self.current_img_tick % len(BIRD_IMG)], self.rot
        )
        win.blit(rotated_image, (self.x, self.y))
        self.current_img_tick += 1

    def get_rects(self):
        """Get bird hitbox [left, top, width, height]"""
        return pygame.Rect(
            self.x,
            self.y,
            BIRD_IMG[self.current_img_tick % len(BIRD_IMG)].get_width(),
            BIRD_IMG[self.current_img_tick % len(BIRD_IMG)].get_height(),
        )


class PIPE:
    PIPE_IMG = pygame.image.load(os.path.join("imgs", "pipe.png"))
    PIPE_WIDTH = PIPE_IMG.get_width()
    PIPE_HEIGHT = PIPE_IMG.get_height()
    PIPE_GAP = 100
    PIPE_RANGE = -150

    def __init__(self):
        self.x = SCREEN_WIDTH  # Start offscreen
        self.y_top = random.randint(
            self.PIPE_HEIGHT - SCREEN_WIDTH, self.PIPE_RANGE
        )  # Random vertical position for top pipe
        self.y_bottom = (
            self.y_top + self.PIPE_HEIGHT + self.PIPE_GAP
        )  # Bottom pipe is placed after the gap
        self.vel = 5
        self.scored = False

    def move(self):
        """Move pipes to the left"""
        self.x -= self.vel

    def draw(self, win):
        """Draw both pipes (top and bottom) with opposite rotation"""
        # Top pipe (flipped upside down)
        top_pipe = pygame.transform.rotate(self.PIPE_IMG, 180)
        win.blit(top_pipe, (self.x, self.y_top))  # Draw top pipe

        # Bottom pipe
        bottom_pipe = self.PIPE_IMG
        win.blit(bottom_pipe, (self.x, self.y_bottom))

    def get_rects(self):
        """Get hitboxes for both pipes"""
        top_pipe_rect = pygame.Rect(
            self.x, self.y_top, self.PIPE_WIDTH, self.PIPE_HEIGHT
        )
        bottom_pipe_rect = pygame.Rect(
            self.x, self.y_bottom, self.PIPE_WIDTH, self.PIPE_HEIGHT
        )
        return top_pipe_rect, bottom_pipe_rect

    def off_screen(self):
        return self.x + self.PIPE_WIDTH < 0


bird = BIRD()
pipes = []

PIPE_SPAWN_TIME = 1200  # milisecond
last_spawn_time = pygame.time.get_ticks()

score = 0

stop = False
# For display score
pygame.font.init()
score_font = pygame.font.SysFont("Comic Sans MS", 30)

running = True
while running:
    clock.tick(60)

    # Main event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird.jump()

    if not stop:
        # Move bird
        bird.move()

        # Generate pipe
        current_time = pygame.time.get_ticks()
        if current_time - last_spawn_time > PIPE_SPAWN_TIME:
            pipes.append(PIPE())
            last_spawn_time = current_time

        # Move pipes
        for pipe in pipes:
            pipe.move()
            if pipe.off_screen():
                pipes.remove(pipe)

        # Calculate score
        bird_rect = bird.get_rects()
        for pipe in pipes:
            top_rect, bottom_rect = pipe.get_rects()

            # Add score
            if bird_rect[0] > top_rect[0] and not pipe.scored:
                score += 1
                pipe.scored = True

            # Game over condition
            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                stop = True
        
        # Also gameover if it touch the ground
        if bird.y + bird_rect[3] >= base_y:
            stop = True

        # Draw screen
        win.blit(BG_IMG, (bg_x, bg_y))
        for pipe in pipes:
            pipe.draw(win)
        win.blit(BASE_IMG, (base_x, base_y))
        bird.draw(win=win)
        score_surface = score_font.render(f"{score}", False, (0, 0, 0))
        win.blit(score_surface, (0, 0))
        pygame.display.flip()
