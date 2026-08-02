# play_ai.py
import torch, time, argparse, sys
from env import PacEnv, ACTION_SPACE
from dqn_agent import DQNAgent
import pygame
import socketio

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

# If streaming, likely purely headless or server side, 
# but currently PacEnv(render=True) tries to open a window.
# We should allow render=False if streaming.
env = PacEnv(render=not args.stream)
state = env.reset()

# Ensure we use the headless mode if requested
if args.stream:
    env.game.headless = True

agent = DQNAgent(state.shape, ACTION_SPACE)
try:
    agent.q.load_state_dict(torch.load('models/dqn_pacman.pt', map_location=agent.device))
    print("Loaded model from models/dqn_pacman.pt")
except (RuntimeError, FileNotFoundError) as e:
    print(f"Warning: Could not load model: {e}")
    # Note: If this fails during a demo, it might be confusing. 
    # But usually creating a random agent is a fallback.
    print("Using Random Agent.")

agent.q.eval()

done=False
while not done:
    if not args.stream:
        pygame.event.get()
        
    a = agent.act(state, eps=0.05) # Low eps for play mode
    
    state,_,done,_ = env.step(a)
    
    if args.stream and sio:
        time.sleep(0.05) # 20 FPS
        try:
            state_json = env.game.get_json_state()
            sio.emit('worker_update', state_json)
        except Exception as e:
            print(f"Emit Error: {e}")
    elif not args.stream:
        time.sleep(0.05)

