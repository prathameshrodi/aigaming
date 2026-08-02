# pac_env.py
import numpy as np
from collections import deque
from game import Game

ACTION_SPACE = 4              # up, down, left, right

class PacEnv:
    """
    Minimal Gym-like wrapper: reset() -> state
                               step(a) -> state, reward, done, info
    State = (H,W,C) uint8 tensor of:
        0 walls, 1 pellets, 2 Pac-Man, 3 ghosts
    """
    def __init__(self, render=False):
        self.render = render
        self.game   = Game(render=render)
        self.stack_size = 4
        self.frames = deque(maxlen=self.stack_size)

    # ----- API ---------------------------------------------------------------
    def reset(self):
        self.game.reset()
        s = self._get_state()
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(s)
        return self._get_stacked_state()

    def step(self, action:int):
        self.game.step(action)
        s   = self._get_state()
        self.frames.append(s)
        
        stacked_s = self._get_stacked_state()
        r   = self._reward()
        done = self.game.game_over
        return stacked_s, r, done, {}

    # ----- internal helpers --------------------------------------------------
    def _get_stacked_state(self):
        # Concatenate frames along channel dimension
        return np.concatenate(self.frames, axis=2)

    def _get_state(self):
        h = len(self.game.maze.grid)
        w = len(self.game.maze.grid[0])
        state = np.zeros((h, w, 4), dtype=np.uint8)

        # channel 0: walls
        state[:,:,0] = (self.game.maze.grid == 1)

        # channel 1: pellets
        for x,y in self.game.maze.pellets:
            state[y,x,1] = 1

        # channel 2: Pac-Man
        state[self.game.player.y, self.game.player.x, 2] = 1

        # channel 3: ghosts
        for g in self.game.ghosts:
            state[g.y, g.x, 3] = 1
        return state

    def _reward(self):
        r = 10 * self.game.pellets_eaten_this_step if self.game.pellets_eaten_this_step else 0
        if self.game.player_dead:
            pass # User requested removal of -500 penalty
            # r -= 500
        if self.game.win:
            r += 500
        r -= 1.0  # time penalty increased from 0.1
        return r
