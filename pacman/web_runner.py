
import time
import requests
import socketio
import torch
import numpy as np
from game import Game
from dqn_agent import DQNAgent
from env import PacEnv, ACTION_SPACE
from settings import *

# Connect to the Flask-SocketIO server
sio = socketio.Client()

def main():
    try:
        sio.connect('http://localhost:5000')
        print("Connected to Web Server")
    except Exception as e:
        print(f"Could not connect to server: {e}")
        return

    # Initialize Game (Headless)
    env = PacEnv(render=False) # Headless
    # Access inner game for state extraction
    game = env.game
    game.headless = True 

    # Load Agent
    state = env.reset()
    agent = DQNAgent(state.shape, ACTION_SPACE)
    
    try:
        agent.q.load_state_dict(torch.load('models/dqn_pacman.pt', map_location=agent.device))
        print("Model loaded.")
    except Exception as e:
        print(f"Model load failed: {e}. Using random agent.")

    agent.q.eval()
    
    done = False
    while not done:
        # AI Step
        a = agent.act(state, eps=0.05)
        next_state, _, done, _ = env.step(a)
        state = next_state
        
        # Extract JSON state
        json_state = game.get_json_state()
        
        # Emit to server
        sio.emit('worker_update', json_state)
        # OR just rely on the server being the runner? 
        # Actually server.py is just a bridge. This script is the "producer".
        # We need to emit to the server so it can broadcast to clients.
        # But server.py defined 'update_state' internally. 
        # Let's use a simple socketio message.
        
        # Note: server.py needs to listen for this 'provider_update' or similar
        # But wait, server.py above didn't have a listener for incoming state.
        # Let's fix server.py to listen for 'update_from_worker' and broadcast.
        
        time.sleep(0.05) # 20 FPS

if __name__ == '__main__':
    main()
