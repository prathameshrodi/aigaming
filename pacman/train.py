# train.py
import numpy as np, tqdm, torch, os
import socketio
import argparse
import time
from env import PacEnv, ACTION_SPACE
from dqn_agent import DQNAgent
import pygame

# train.py
import numpy as np, tqdm, torch, os
import socketio
import argparse
import time
from env import PacEnv, ACTION_SPACE
from dqn_agent import DQNAgent
import pygame

# --- Parallel Training Config ---
NUM_ENVS = 8  # "Generations"/"Multiple Players" - Speed up collection
STREAM_ENV_IDX = 0 # Only stream the first environment

parser = argparse.ArgumentParser()
parser.add_argument('--stream', action='store_true', help='Stream to localhost:5000')
args = parser.parse_args()

sio = None
if args.stream:
    sio = socketio.Client()
    try:
        sio.connect('http://localhost:5000')
        print("Connected to Stream Server")
    except Exception as e:
        print(f"Server connection failed: {e}")
        sio = None

# Initialize Envs
envs = [PacEnv(render=False) for _ in range(NUM_ENVS)]
if args.stream:
    envs[STREAM_ENV_IDX].game.headless = True 

agent = DQNAgent(envs[0].reset().shape, ACTION_SPACE)

episodes       = 1000
eps_start      = 1.0
eps_end        = 0.01
eps_decay_rate = 0.9990

# We track state for each env
states = [e.reset() for e in envs]
scores = [0] * NUM_ENVS
dones  = [False] * NUM_ENVS
eps    = eps_start

# Main Loop (Step 1 by 1 for all envs)
# Since environments can have different lengths, we treat "episodes" as "batches of steps" or count finished games.
# For simplicity, we loop for a fixed number of total steps or until X episodes finish (globally).
total_steps = 1000000 
completed_episodes = 0

pbar = tqdm.tqdm(total=episodes)

print(f"Starting Parallel Training with {NUM_ENVS} players...")

while completed_episodes < episodes:
    
    if not args.stream:
         pygame.event.get() 

    # 1. Select Actions for all envs
    # Agent.act handles single state, let's just loop (simple) or batch if act supported it.
    # Current agent.act expects single state.
    actions = [agent.act(s, eps) for s in states]

    # 2. Step all envs
    for i in range(NUM_ENVS):
        s2, r, done, _ = envs[i].step(actions[i])
        
        # Stream update (only for first env)
        if args.stream and sio and i == STREAM_ENV_IDX: 
             time.sleep(0.01) # Limit FPS slightly so we can see it
             try:
                state_json = envs[i].game.get_json_state()
                sio.emit('worker_update', state_json)
             except Exception as e:
                print(f"Emit Error: {e}")

        # Store transition
        agent.remember(states[i], actions[i], r, s2, done)
        
        start_state = s2
        if done:
            # Stats
            if i == STREAM_ENV_IDX and sio:
                 sio.emit('training_log', {'episode': completed_episodes, 'score': scores[i], 'epsilon': eps})
            
            # Reset this env
            start_state = envs[i].reset()
            scores[i] = 0
            completed_episodes += 1
            pbar.update(1)
            
            # decay epsilon globally (or per agent?) Global is fine for "Generations" feel
            eps = max(eps_end, eps * eps_decay_rate)
            
        states[i] = start_state
        scores[i] += r

    # 3. Train (once per batch of steps)
    agent.train_step()

    if completed_episodes % 10 == 0:
        pbar.set_description(f"Eps: {eps:.2f} Score: {np.mean(scores):.1f}")

    # Auto Save
    if completed_episodes % 50 == 0:
        os.makedirs('models', exist_ok=True)
        torch.save(agent.q.state_dict(), 'models/dqn_pacman.pt')

print("Model saved to models/dqn_pacman.pt")
torch.save(agent.q.state_dict(), 'models/dqn_pacman.pt')


