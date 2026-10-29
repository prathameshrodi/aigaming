import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Standard 10x10 Maze Layout with inner walls
# 0 = Open Walkable Path
# 1 = Wall Barrier
MAZE_LAYOUT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Y=9 (Top)
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],  # Y=8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Y=7
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],  # Y=6
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],  # Y=5
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 0],  # Y=4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Y=3
    [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],  # Y=2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Y=1
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Y=0 (Bottom)
], dtype=np.int8)

class Pac2DEnv(gym.Env):
    """
    Standard 2D Grid Game with Maze Walls for AI Training.
    
    Objective:
      Find the fastest path to the target (Core) while evading the enemy (Ghost)
      and navigating around impassable walls.
      
    Action Space (Discrete 5):
      0: Idle / Stay
      1: Move Left  (X - 1)
      2: Move Right (X + 1)
      3: Move Up    (Y + 1)
      4: Move Down  (Y - 1)
      
    Observation Space (Box 6):
      [Player_X, Player_Y, Enemy_X, Enemy_Y, Target_X, Target_Y]
      Normalized to continuous float32 values [0.0, 9.0].
    """
    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4.0}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.grid_width = 10
        self.grid_height = 10
        self.max_steps = 60

        self.maze = MAZE_LAYOUT.copy()

        # 5 actions: Idle, Left, Right, Up, Down
        self.action_space = spaces.Discrete(5)

        # 6 continuous coordinate values: [P_x, P_y, E_x, E_y, T_x, T_y]
        low = np.zeros(6, dtype=np.float32)
        high = np.full(6, 9.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Precompute all open (walkable) coordinates
        self.open_cells = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not self.is_wall(x, y):
                    self.open_cells.append((x, y))

        self.player_pos = np.array([0, 0], dtype=np.int32)
        self.enemy_pos = np.array([0, 0], dtype=np.int32)
        self.target_pos = np.array([0, 0], dtype=np.int32)
        self.step_count = 0

    def is_wall(self, x, y):
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        # MAZE_LAYOUT row 0 corresponds to Y=9
        row = (self.grid_height - 1) - y
        col = x
        return self.maze[row, col] == 1

    def _get_obs(self):
        return np.array([
            float(self.player_pos[0]),
            float(self.player_pos[1]),
            float(self.enemy_pos[0]),
            float(self.enemy_pos[1]),
            float(self.target_pos[0]),
            float(self.target_pos[1])
        ], dtype=np.float32)

    def _get_info(self):
        dist_target = abs(self.player_pos[0] - self.target_pos[0]) + abs(self.player_pos[1] - self.target_pos[1])
        dist_enemy = abs(self.player_pos[0] - self.enemy_pos[0]) + abs(self.player_pos[1] - self.enemy_pos[1])
        return {
            "step": self.step_count,
            "dist_target": dist_target,
            "dist_enemy": dist_enemy
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Spawn Player, Enemy, and Target in distinct open (non-wall) cells
        rng = self.np_random
        chosen_indices = rng.choice(len(self.open_cells), size=3, replace=False)

        self.player_pos = np.array(self.open_cells[chosen_indices[0]], dtype=np.int32)
        self.enemy_pos = np.array(self.open_cells[chosen_indices[1]], dtype=np.int32)
        self.target_pos = np.array(self.open_cells[chosen_indices[2]], dtype=np.int32)

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        self.step_count += 1

        prev_dist_target = abs(self.player_pos[0] - self.target_pos[0]) + abs(self.player_pos[1] - self.target_pos[1])

        # 1. Player Intentional 2D Movement with Wall Blocking
        new_x, new_y = self.player_pos[0], self.player_pos[1]
        bumped_wall = False

        if action == 1:    # Left
            new_x -= 1
        elif action == 2:  # Right
            new_x += 1
        elif action == 3:  # Up
            new_y += 1
        elif action == 4:  # Down
            new_y -= 1
        # action == 0: Idle / Stay

        if action != 0:
            if self.is_wall(new_x, new_y):
                bumped_wall = True
            else:
                self.player_pos[0] = new_x
                self.player_pos[1] = new_y

        # 2. Enemy Navigation (Chases player around walls)
        self._move_enemy()

        # 3. Compute Reward & Terminal states
        terminated = False
        truncated = False

        new_dist_target = abs(self.player_pos[0] - self.target_pos[0]) + abs(self.player_pos[1] - self.target_pos[1])
        dist_enemy = abs(self.player_pos[0] - self.enemy_pos[0]) + abs(self.player_pos[1] - self.enemy_pos[1])

        # Base step penalty (encourages finding the fastest path)
        reward = -0.2

        # Small distance-shaping reward: reward getting closer to target, penalize moving away
        if new_dist_target < prev_dist_target:
            reward += 0.5
        elif new_dist_target > prev_dist_target:
            reward -= 0.5

        # Penalty for bumping into walls
        if bumped_wall:
            reward -= 1.0

        # Danger zone penalty: evading enemy
        if dist_enemy <= 1:
            reward -= 3.0
        elif dist_enemy == 2:
            reward -= 1.0

        # Terminal conditions
        if np.array_equal(self.player_pos, self.enemy_pos):
            reward = -100.0
            terminated = True
        elif np.array_equal(self.player_pos, self.target_pos):
            reward = 100.0
            terminated = True
        elif self.step_count >= self.max_steps:
            reward = 0.0
            truncated = True

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _move_enemy(self):
        """
        Enemy selects the best legal (non-wall) move that reduces Manhattan distance to Player.
        """
        ex, ey = self.enemy_pos[0], self.enemy_pos[1]
        px, py = self.player_pos[0], self.player_pos[1]

        candidates = [(ex, ey), (ex - 1, ey), (ex + 1, ey), (ex, ey + 1), (ex, ey - 1)]
        best_pos = (ex, ey)
        min_dist = float('inf')

        for cx, cy in candidates:
            if not self.is_wall(cx, cy):
                d = abs(cx - px) + abs(cy - py)
                if d < min_dist:
                    min_dist = d
                    best_pos = (cx, cy)

        self.enemy_pos[0] = best_pos[0]
        self.enemy_pos[1] = best_pos[1]

    def render(self):
        grid = []
        for y in range(self.grid_height - 1, -1, -1):
            row = []
            for x in range(self.grid_width):
                if self.is_wall(x, y):
                    row.append(" # ")
                elif x == self.target_pos[0] and y == self.target_pos[1]:
                    row.append(" T ")
                elif x == self.enemy_pos[0] and y == self.enemy_pos[1]:
                    row.append(" E ")
                elif x == self.player_pos[0] and y == self.player_pos[1]:
                    row.append(" P ")
                else:
                    row.append(" . ")
            grid.append(row)

        lines = [f"\n--- [2D PAC-NAV] Step: {self.step_count}/{self.max_steps} ---"]
        lines.append("    +" + "---" * self.grid_width + "+")
        for i, row in enumerate(grid):
            y_coord = (self.grid_height - 1) - i
            lines.append(f" {y_coord}  |{''.join(row)}|")
        lines.append("    +" + "---" * self.grid_width + "+")
        lines.append("       " + "".join([f" {x} " for x in range(self.grid_width)]))
        lines.append(f"Player: {self.player_pos.tolist()} | Enemy: {self.enemy_pos.tolist()} | Target: {self.target_pos.tolist()}\n")

        output = "\n".join(lines)
        if self.render_mode == "ansi":
            return output
        else:
            print(output)
            return output
