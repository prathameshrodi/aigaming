# ghost.py
import pygame, random, pygame.time
from settings import PURPLE, TILE_SIZE, RED, WHITE

class Ghost:
    """
    Random-walking ghost that respects maze walls and moves on a cooldown.
    Arguments
    ---------
    maze : Maze           – reference to the Maze object for wall checks
    idx  : int            – index so multiple ghosts can spawn differently
    color: (r,g,b) tuple  – customise ghost colour if desired
    """
    # Valid spawn positions (checked against layout)
    # (1, 1) is usually safe (top left corner inner)
    # (13, 11) is middle-ish
    # (1, 29) bottom left
    # (26, 1) top right
    SPAWN_POSITIONS = [(1, 1), (13, 11), (1, 29), (26, 1), (26, 29), (13, 23)]  
    COLOR = [RED, WHITE, PURPLE, (0, 255, 0), (255, 165, 0), (0, 0, 255)] # Red, White, Purple, Green, Orange, Blue

    def __init__(self, maze, idx=0, color=RED, move_delay_steps=15):
        self.maze       = maze
        self.x, self.y  = Ghost.SPAWN_POSITIONS[idx % len(self.SPAWN_POSITIONS)]
        self.color      = Ghost.COLOR[idx % len(Ghost.COLOR)]
        self.move_delay_steps = move_delay_steps
        self.step_counter = 0

    # ───────────────────────────────────────────────────────────────── update ──
    def move(self, player):
        self.step_counter += 1
        if self.step_counter % self.move_delay_steps == 0:
            self._move_random()

        # Collision with Pac-Man?
        if self.x == player.x and self.y == player.y:
            player.alive = False

    # ──────────────────────────────────────────────────────────── movement AI ──
    def _move_random(self):
        dirs = [(0,-1),(0,1),(-1,0),(1,0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < self.maze.width and 0 <= ny < self.maze.height:
                if not self.maze.is_wall(nx, ny):
                    self.x, self.y = nx, ny
                    break

    # ───────────────────────────────────────────────────────────────── draw ──
    def draw(self, screen):
        pygame.draw.circle(
            screen,
            self.color,
            (self.x*TILE_SIZE + TILE_SIZE//2,
             self.y*TILE_SIZE + TILE_SIZE//2),
            TILE_SIZE//2
        )
