import os
import pygame
from maze import Maze
from player import Player
from ghost import Ghost
from settings import *

    

class Game:
    def __init__(self, render=True, headless=False):
        pygame.init()
        self.pellets_eaten_this_step = 0
        self.render = render and not headless
        self.headless = headless
        if self.render:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = None
            
        self.clock = pygame.time.Clock()
        self.exiting_loc = (0,0)
        self.reset()

    def reset(self):
        self.maze = Maze()
        self.player = Player(self.maze)
        # Spawn all ghosts defined in Ghost class
        self.ghosts = [
            Ghost(self.maze, i) for i in range(len(Ghost.SPAWN_POSITIONS))
        ]
        self.game_over = False
        self.win = False
        self.pellets_eaten_this_step = 0
        self.player_dead = False

        if self.render:
            pygame.display.set_caption("Pac-Man AI")
            self.screen.fill(BLACK)
            self.maze.draw(self.screen)
            pygame.display.flip()
   


    def step(self, action):  # action: 0=up 1=down 2=left 3=right
        if self.game_over:
            return

        dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
        self.player.move(dx, dy)
        self.pellets_eaten_this_step = self.player.consume_pellet(dx, dy)

        for ghost in self.ghosts:
            ghost.move(self.player)

        if self._check_collision():
            self.game_over = True
            self.player_dead = True

        if self.player.score >= self.maze.total_pellets:
            self.game_over = True
            self.win = True

        if self.render:
            self._render()

    def _check_collision(self):
        for ghost in self.ghosts:
            if ghost.x == self.player.x and ghost.y == self.player.y:
                return True
        return False

    def _render(self):
        self.screen.fill(BLACK)
        self.maze.draw(self.screen)
        self.player.draw(self.screen)
        for ghost in self.ghosts:
            ghost.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def get_json_state(self):
        return {
            "player": {"x": int(self.player.x), "y": int(self.player.y)},
            "ghosts": [{"x": int(g.x), "y": int(g.y), "color": [int(c) for c in g.color]} for g in self.ghosts],
            "pellets": [{"x": int(x), "y": int(y)} for x, y in self.maze.pellets],
            "walls": self.maze.grid, 
            "score": int(self.player.score),
            "game_over": bool(self.game_over),
            "win": bool(self.win),
            "total_pellets": int(self.maze.total_pellets)
        }
    
    def final_score(self):
        return self.player.score
