# maze.py
import pygame
from settings import TILE_SIZE, BLUE

_LAYOUT = [
    "1111111111111111111111111111",
    "1000000000110000000000110001",
    "1011111110110111111110110101",
    "1011111110110111111110110101",
    "1011111110110111111110110101",
    "1000000000000000000000000001",
    "1011110111111111111101111101",
    "1000010110000000001101000011",
    "1111010110111111101101010111",
    "1111010110100000101101010111",
    "1111010110101110101101010111",
    "1000000000101110100000000001",
    "1011111111101110111111111101",
    "1000000000000000000000000001",
    "1111010111111111111101010111",
    "1111010100000000000101010111",
    "1111010101111111110101010111",
    "1000010001000000001000010001",
    "1011111110110111110111111101",
    "1000000010000100000100000001",
    "1111110111110111111101111111",
    "1111110100000000000101111111",
    "1111110101111111110101111111",
    "1000000101000000010100000001",
    "1011111101011111010111111101",
    "1000000001000000010000000001",
    "1111011111110111111111101111",
    "1000010000000000000000010001",
    "1011110111110111111111011101",
    "1000000000110000000011000001",
    "1111111111111111111111111111",
    "1111111111111111111111111111"
]

# _LAYOUT = ["111111",
#            "100001",
#            "101101",
#            "101101",
#            "100001",
#            "111111"]
class Maze:
    """Walls + pellets + drawing utilities"""
    def __init__(self):
        # Convert string rows to int grid: 1 wall, 0 path
        self.grid = [[1 if ch == '1' else 0 for ch in row] for row in _LAYOUT]
        self._init_pellets()
        self.width = len(self.grid[0])
        self.height = len(self.grid)

    # --------------------------------------------------------------------- Pellets
    def _init_pellets(self):
        self.pellets = {(x, y)
                        for y, row in enumerate(self.grid)
                        for x, cell in enumerate(row) if cell == 0}
        self.total_pellets = len(self.pellets)

    def has_pellet(self, x, y):
        return (x, y) in self.pellets

    def eat_pellet(self, x, y):
        """Returns True if a pellet was eaten and removes it"""
        return self.pellets.discard((x, y)) is None  # True if present

    # --------------------------------------------------------------------- Walls
    def is_wall(self, x, y):
        return self.grid[y][x] == 1

    # --------------------------------------------------------------------- Drawing
    def draw(self, screen):
        # Draw walls
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == 1:
                    rect = pygame.Rect(x*TILE_SIZE, y*TILE_SIZE,
                                       TILE_SIZE, TILE_SIZE)
                    pygame.draw.rect(screen, BLUE, rect)

        # Draw pellets
        for x, y in self.pellets:
            pygame.draw.circle(screen,
                               (255, 255, 255),
                               (x*TILE_SIZE + TILE_SIZE//2,
                                y*TILE_SIZE + TILE_SIZE//2),
                               3)
    
    def hide_pellet(self, x, y):
        """Hides a pellet at the given coordinates"""
        if (x, y) in self.pellets:
            self.pellets.remove((x, y))