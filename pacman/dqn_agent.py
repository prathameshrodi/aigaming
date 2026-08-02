# dqn_agent.py
import torch, torch.nn as nn, torch.optim as optim, random, numpy as np
from collections import deque
from torch.nn import LazyLinear

class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            LazyLinear(512), nn.ReLU(),   # <-- input dim auto-detects
            nn.Linear(512, n_actions)
        )


    def forward(self, x):  # x (B,C,H,W)
        return self.net(x)

class DQNAgent:
    def __init__(self, state_shape, n_actions,
                 gamma=0.99, lr=1e-4, buffer_size=50_000):
        c = state_shape[2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q         = DQN(c, n_actions).to(self.device)
        self.q_target  = DQN(c, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt       = optim.Adam(self.q.parameters(), lr=lr)
        self.gamma     = gamma
        self.replay    = deque(maxlen=buffer_size)
        self.batch     = 64
        self.sync_every= 1000
        self.steps     = 0
        self.n_actions = n_actions

    # ε-greedy
    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).permute(0,3,1,2).float().to(self.device)
            q = self.q(s)
            return int(torch.argmax(q))

    def remember(self, s,a,r,s2,d):
        self.replay.append((s,a,r,s2,d))

    def train_step(self):
        if len(self.replay) < self.batch: return
        batch = random.sample(self.replay, self.batch)
        s,a,r,s2,d = map(np.array, zip(*batch))

        s  = torch.tensor(s).float().to(self.device)
        s2 = torch.tensor(s2).float().to(self.device)
        s = s.permute(0, 3, 1, 2)
        s2 = s2.permute(0, 3, 1, 2)
        a  = torch.tensor(a).to(self.device)
        r  = torch.tensor(r).to(self.device)
        d  = torch.tensor(d).float().to(self.device)

        q_vals = self.q(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            max_next = self.q_target(s2).max(1)[0]
            target   = r + self.gamma * max_next * (1 - d)
        
        target = torch.tensor(target, dtype=torch.float)
        loss = nn.functional.mse_loss(q_vals, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        self.steps += 1
        if self.steps % self.sync_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())
