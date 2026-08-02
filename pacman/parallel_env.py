# parallel_env.py
import multiprocessing as mp
import numpy as np
from game import Game          # your current pygame loop
from settings import FPS

def worker(id, state_q, cmd_q):
    env = Game(headless=True)   # add 'headless' flag so pygame uses dummy video
    while True:
        cmd = cmd_q.get()
        if cmd == "reset":
            s = env.reset()
            state_q.put((id, s))
        elif isinstance(cmd, int):             # action
            s, r, done, _ = env.step(cmd)
            state_q.put((id, s, r, done))
            if done: env.reset()
        elif cmd == "close":
            env.close(); break

class ParallelPacman:
    def __init__(self, n_envs=8):
        self.n_envs = n_envs
        ctx = mp.get_context("spawn")
        self.state_q = ctx.Queue()
        self.workers  = []
        self.cmd_qs   = []
        for i in range(n_envs):
            cmd_q = ctx.Queue()
            p = ctx.Process(target=worker, args=(i, self.state_q, cmd_q))
            p.daemon = True; p.start()
            self.workers.append(p); self.cmd_qs.append(cmd_q)
            cmd_q.put("reset")

    # ── vector API ───────────────────────────────────────────
    def step_async(self, actions):
        for q, a in zip(self.cmd_qs, actions):
            q.put(int(a))

    def step_wait(self):
        obs, rews, dones = [None]*self.n_envs, [0.0]*self.n_envs, [False]*self.n_envs
        for _ in range(self.n_envs):
            id, *data = self.state_q.get()
            if len(data) == 1:                 # just obs after reset
                obs[id] = data[0]
            else:
                obs[id], rews[id], dones[id] = data
        return np.stack(obs), np.array(rews), np.array(dones)

    def close(self):
        for q in self.cmd_qs: q.put("close")
        for p in self.workers: p.join()

# usage inside train loop  ───────────────────────────────────

