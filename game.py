import ctypes
import os
import math

# --- Windows DPI Fix (Must be at the very top of the file) ---
try:
    # Modern Windows 10/11 approach
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # 2 = Per Monitor DPI Aware
except Exception:
    try:
        # Fallback for older Windows versions
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Optional: Prevents Pygame from messing with the window if you click away
os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
# -------------------------------------------------------------

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', ['x', 'y'])

# rgb colors
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 40 # for (32, 24) board
SPEED = 175 #75

class SnakeGameAi:
    def __init__(self, w=1280, h=960):
        self.w = w
        self.h = h
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

        self.frame_iteration = 0 # game iteration (t, time)
        self.reset()
                
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def play_step(self, action):
        self.frame_iteration += 1

        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Dist from the apple before move
        dist_before = math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        # Dist from the apple after move
        dist_after = math.sqrt((self.head.x - self.food.x)**2 + (self.head.y - self.food.y)**2)
        
        # 3. Small reward on move away/move closer to the apple
        reward = 0
        if dist_after < dist_before:
            reward = 0.1  
        else:
            reward = -0.1 

        # 4. check if game over
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 5. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 6. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 7. return game over and score
        return reward, game_over, self.score
    
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        if not pt:
            pt = self.head

        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)

        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GRAY, (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, (GRAY), (0, y), (self.w, y), 1)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x + 1, pt.y + 1, BLOCK_SIZE - 1, BLOCK_SIZE - 1))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x + 1, self.food.y + 1, BLOCK_SIZE - 1, BLOCK_SIZE - 1))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        """
        # action -> [straight, right, left] (boolian values)

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn: right -> down -> left -> up -> right
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn: right -> up -> left -> down -> right 

        self.direction = new_dir
        """
        # action -> [UP, RIGHT, DOWN, LEFT]

        if np.array_equal(action, [1, 0, 0, 0]):
            self.direction = Direction.UP
        elif np.array_equal(action, [0, 1, 0, 0]):
            self.direction = Direction.RIGHT
        elif np.array_equal(action, [0, 0, 1, 0]):
            self.direction = Direction.DOWN
        elif np.array_equal(action, [0, 0, 0, 1]):
            self.direction = Direction.LEFT

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

def get_idx(point):
    # Helper function to convert point (in pixels) to matrix index
    x_idx = int(point.x // BLOCK_SIZE)
    y_idx = int(point.y // BLOCK_SIZE)
    return x_idx, y_idx