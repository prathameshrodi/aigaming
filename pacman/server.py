from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time
import json
import psutil
import os
import signal
import subprocess
import sys

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# Allow all origins for dev simplicity
socketio = SocketIO(app, cors_allowed_origins="*")

# Store the latest game state
latest_state = None
game_thread = None
stop_event = threading.Event()
training_process = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('worker_update')
def handle_worker_update(data):
    # print(".", end="", flush=True) # Simple tick
    socketio.emit('game_state', data)

@socketio.on('training_log')
def handle_training_log(data):
    socketio.emit('training_log', data)

@socketio.on('start_training')
def start_training():
    global training_process
    if training_process and training_process.poll() is None:
        print("Training already running")
        return
    
    print("Starting training process...")
    # Launch train.py with --stream flag
    # Use unbuffered output (-u) to get logs instantly
    cmd = [sys.executable, '-u', 'pacman/train.py', '--stream']
    
    # Capture stdout and stderr
    training_process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        universal_newlines=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # Start a background thread to read logs
    def log_reader(proc):
        for line in iter(proc.stdout.readline, ''):
            if line:
                socketio.emit('server_log', {'msg': line.strip()})
        proc.stdout.close()
        
    t = threading.Thread(target=log_reader, args=(training_process,))
    t.daemon = True
    t.start()

@socketio.on('start_game')
def start_game():
    global training_process
    if training_process and training_process.poll() is None:
        print("Process already running (Game or Training)")
        return
    
    print("Starting Game (Inference)...")
    cmd = [sys.executable, '-u', 'pacman/play_ai.py', '--stream']
    
    training_process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1, 
        universal_newlines=True,
        encoding='utf-8',
        errors='replace'
    )
    
    def log_reader(proc):
        for line in iter(proc.stdout.readline, ''):
            if line:
                socketio.emit('server_log', {'msg': line.strip()})
        proc.stdout.close()
        
    t = threading.Thread(target=log_reader, args=(training_process,))
    t.daemon = True
    t.start()

@socketio.on('stop_training')
def stop_training():
    # Stops any running process (training or game)
    global training_process
    if training_process and training_process.poll() is None:
        print("Stopping training...")
        training_process.terminate()
        try:
            training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            training_process.kill()
        training_process = None
        print("Training stopped.")

@socketio.on('shutdown')
def shutdown():
    print("Shutting down server...")
    os.kill(os.getpid(), signal.SIGINT)

def update_state(state):
    global latest_state
    latest_state = state
    socketio.emit('game_state', state)

def run_server():
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    run_server()
