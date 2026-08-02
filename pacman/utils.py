# utils.py
from collections import deque
from maze import MAZE_LAYOUT

DIRS = [(0,-1),(0,1),(-1,0),(1,0)]

def bfs(start, goals, walls):
    """Return dict: cell -> parent, ending at the first goal reached."""
    q = deque([start])
    parent = {start: None}
    while q:
        x, y = q.popleft()
        if (x, y) in goals:
            return parent, (x, y)          # found nearest goal
        for dx, dy in DIRS:
            nx, ny = x+dx, y+dy
            if (0 <= ny < len(MAZE_LAYOUT) and 0 <= nx < len(MAZE_LAYOUT[0])
                and MAZE_LAYOUT[ny][nx] == '0'
                and (nx, ny) not in walls
                and (nx, ny) not in parent):
                parent[(nx, ny)] = (x, y)
                q.append((nx, ny))
    return None, None                      # no path
