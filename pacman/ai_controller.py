# ai_controller.py
import pygame
from utils import bfs, DIRS
from maze import MAZE_LAYOUT

class PacAI:
    def __init__(self, player, ghosts, pellets):
        self.player  = player
        self.ghosts  = ghosts
        self.pellets = pellets

    def choose_direction(self):
        """Return (dx, dy) each frame."""
        # Build sets for BFS
        pac_pos   = (self.player.x, self.player.y)
        ghost_pos = {(g.x, g.y) for g in self.ghosts}
        if not self.pellets:
            return 0, 0                     # nowhere to go

        # 1. Find nearest pellet using BFS that avoids ghost tiles
        parent, goal = bfs(pac_pos, self.pellets, ghost_pos)
        if goal is None:
            return 0, 0                     # stuck

        # 2. Backtrack one step from goal toward Pac-Man
        cur = goal
        while parent[cur] and parent[cur] != pac_pos:
            cur = parent[cur]
        dx = cur[0] - pac_pos[0]
        dy = cur[1] - pac_pos[1]
        return dx, dy
