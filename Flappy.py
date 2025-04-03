import pygame
import random
import os

# Initialize pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
GRAVITY = 0.5
FLAP_STRENGTH = -8
PIPE_GAP = 100
PIPE_VELOCITY = -4
PIPE_SPAWN_DISTANCE = 300  # Distance between pipes
BASE_VELOCITY = 2  # Speed of base scrolling

def load_image(name):
    return pygame.image.load(os.path.join("assets", "sprites", name))

def load_sound(name):
    return pygame.mixer.Sound(os.path.join("assets", "audio", name))

# Load Assets
BACKGROUND = load_image("background-black.png")
BASE = load_image("base.png")
PIPE = load_image("pipe-green.png")
BIRD_FRAMES = [
    load_image("redbird-downflap.png"),
    load_image("redbird-midflap.png"),
    load_image("redbird-upflap.png"),
]

SOUNDS = {
    "die": load_sound("die.wav"),
    "hit": load_sound("hit.wav"),
    "point": load_sound("point.wav"),
    "swoosh": load_sound("swoosh.wav"),
    "wing": load_sound("wing.wav"),
}

class Bird:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.frame = 0
        self.width = BIRD_FRAMES[0].get_width()
        self.height = BIRD_FRAMES[0].get_height()

    def flap(self):
        self.velocity = FLAP_STRENGTH
        SOUNDS["wing"].play()

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        self.frame = (self.frame + 1) % len(BIRD_FRAMES)

    def draw(self, screen):
        screen.blit(BIRD_FRAMES[self.frame], (self.x, self.y))

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(100, 300)
        self.width = PIPE.get_width()
        self.pipe_height = PIPE.get_height()

    def update(self):
        self.x += PIPE_VELOCITY

    def draw(self, screen):
        # Draw lower pipe
        screen.blit(PIPE, (self.x, self.height + PIPE_GAP))
        # Draw upper pipe (flipped)
        screen.blit(pygame.transform.flip(PIPE, False, True),
                    (self.x, self.height - self.pipe_height))

    def get_rects(self):
        return [
            pygame.Rect(self.x, self.height + PIPE_GAP, self.width, self.pipe_height),  # Lower pipe
            pygame.Rect(self.x, self.height - self.pipe_height, self.width, self.pipe_height)  # Upper pipe
        ]

class FlappyBirdEnv:
    def __init__(self, render_mode=False):
        self.render_mode = render_mode
        self.screen = None
        if self.render_mode:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game state and returns the initial observation."""
        pygame.event.clear()
        self.bird = Bird()
        self.pipes = [Pipe(SCREEN_WIDTH)]
        self.base_x = 0
        self.score = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Returns a state representation.
        [bird_y, bird_velocity, distance_to_next_pipe, pipe_top, pipe_bottom]
        """
        next_pipe = self.pipes[0]
        for pipe in self.pipes:
            if pipe.x + pipe.width > self.bird.x:
                next_pipe = pipe
                break

        return [
            self.bird.y,
            self.bird.velocity,
            next_pipe.x - self.bird.x,
            next_pipe.height,  # Top of gap (upper pipe bottom)
            next_pipe.height + PIPE_GAP  # Bottom of gap (lower pipe top)
        ]

    def step(self, action):
        """
        Performs one game step.
        action: 0 for no action, 1 for flap.
        Returns: observation, reward, done, info
        """
        if action == 1:
            self.bird.flap()

        # Update game state
        self.bird.update()
        for pipe in self.pipes:
            pipe.update()

        # Pipe management
        if self.pipes[0].x < -self.pipes[0].width:
            self.pipes.pop(0)
            self.score += 1
            SOUNDS["point"].play()

        if len(self.pipes) < 2 and self.pipes[-1].x < SCREEN_WIDTH - PIPE_SPAWN_DISTANCE:
            self.pipes.append(Pipe(SCREEN_WIDTH))

        # **Fixed Base Scrolling**
        self.base_x = (self.base_x - BASE_VELOCITY) % (-BASE.get_width())

        # Check for collisions
        reward = 0.1  # Small reward for staying alive
        if (self.bird.y < 0 or
                self.bird.y + self.bird.height > SCREEN_HEIGHT - BASE.get_height()):
            SOUNDS["die"].play()
            reward = -1
            self.done = True

        for pipe in self.pipes:
            for rect in pipe.get_rects():
                if self.bird.get_rect().colliderect(rect):
                    SOUNDS["hit"].play()
                    reward = -1
                    self.done = True

        observation = self.get_state()
        info = {"score": self.score}

        if self.render_mode:
            self.render()

        return observation, reward, self.done, info

    def render(self):
        """Renders the current game state."""
        self.screen.blit(BACKGROUND, (0, 0))

        for pipe in self.pipes:
            pipe.draw(self.screen)

        self.bird.draw(self.screen)

        # **Fixed Base Rendering**
        self.screen.blit(BASE, (self.base_x, SCREEN_HEIGHT - BASE.get_height()))
        self.screen.blit(BASE, (self.base_x + BASE.get_width(), SCREEN_HEIGHT - BASE.get_height()))

        pygame.display.update()
        self.clock.tick(30)

def interactive_main():
    """Main loop for interactive play."""
    env = FlappyBirdEnv(render_mode=True)
    running = True

    while running:
        action = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                action = 1

        _, _, done, _ = env.step(action)

        if done:
            pygame.time.delay(1000)  # Pause before closing
            running = False

if __name__ == "__main__":
    interactive_main()
    pygame.quit()
