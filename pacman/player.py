# player.py

import pygame
from settings import TILE_SIZE, YELLOW, WIDTH, HEIGHT, BLACK
from maze import _LAYOUT as MAZE_LAYOUT

class Player:
    def __init__(self, maze):
        self.x = 1
        self.y = 1
        self.move_delay = 150  # milliseconds between moves
        self.last_move_time = 0
        self.maze = maze
        self.score = 1

    def draw(self, screen):
        pygame.draw.circle(
            screen,
            YELLOW,
            (self.x * TILE_SIZE + TILE_SIZE // 2, self.y * TILE_SIZE + TILE_SIZE // 2),
            TILE_SIZE // 2 - 2
        )

    def move(self, dx, dy):
        current_time = pygame.time.get_ticks()
        new_x = self.x + dx
        new_y = self.y + dy

        # ✅ Prevent out-of-bounds access
        if 0 <= new_y < len(MAZE_LAYOUT) and 0 <= new_x < len(MAZE_LAYOUT[0]):
            if MAZE_LAYOUT[new_y][new_x] == '0':
                self.x = new_x
                self.y = new_y
                self.last_move_time = current_time

    def consume_pellet(self, dx, dy):
        new_x = self.x + dx
        new_y = self.y + dy
        pellets_eaten = 0  # Initialize the number of pellets eaten

        if not self.maze.is_wall(new_x, new_y):
            # self.x, self.y = dx, dy  # Incorrect: This sets the position to dx, dy
            #Correct: Update player position
            # self.x = new_x
            # self.y = new_y
            if self.maze.has_pellet(new_x, new_y):
                self.maze.eat_pellet(new_x, new_y)
                self.maze.hide_pellet(new_x, new_y)
                self.score += 1
                pellets_eaten = 1  # Increment pellets eaten

        return pellets_eaten  # Return the number of pellets eaten
