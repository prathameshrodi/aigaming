# 👾 PAC-MAN 2D & ANTIGRAVITY: AI Reinforcement Learning & Imitation Lab

A dual-mode Deep Reinforcement Learning (DRL) and Behavioral Cloning research platform built with **Gymnasium**, **PyTorch**, and **Stable-Baselines3 (PPO)**, paired with a modern, single-file retro **Pac-Man Arcade Web UI**.

Repository: [https://github.com/prathameshrodi/aigaming.git](https://github.com/prathameshrodi/aigaming.git)

---

## 🌌 Project Overview

This repository provides two complete AI training architectures and an interactive simulation laboratory:

1. **Standard 2D Pac-Man Navigation with Dynamic Maze Walls**:
   - The agent learns the **fastest path** around procedural walls to reach the Target while actively **evading Blinky** (the chasing ghost).
   - **Dynamic Mazes**: Every time a level is cleared, the maze re-creates its wall architecture, forcing the AI to generalize rather than memorize fixed corridors.
2. **Behavioral Cloning (Human Demonstration Recorder)**:
   - Play manually in the browser using the D-Pad / Arrow keys.
   - Record exact state-action transitions to export `human_play_data.json`.
   - Pretrain a policy network with supervised imitation learning, followed by PPO fine-tuning.
3. **Antigravity (Physics-Inverted Underactuated Grid Sandbox)**:
   - An experimental reinforcement learning benchmark where the player has **no vertical thrusters**.
   - Movement across the vertical axis requires riding a periodic, inverted external gravity vector ($\pm 1$) that reverses every 8 ticks.

---

## 📁 Repository File Architecture

```text
aigaming/
├── environment.py         # Gymnasium Environment (Pac2DEnv) with maze collision & enemy evasion
├── train.py               # PPO training pipeline using Stable-Baselines3 (RL from scratch)
├── train_from_human.py    # Hybrid training: Phase 1 Behavioral Cloning + Phase 2 PPO Fine-Tuning
├── enjoy.py               # Deterministic terminal evaluation with real-time ASCII rendering
├── index.html             # Single-file retro arcade Web UI, audio synth & demonstration recorder
├── .gitignore             # Ignores .venv, cache, and OS artifacts
└── README.md              # Comprehensive documentation & mathematical specification
```

---

## 🧱 1. Reinforcement Learning Environment Specification

### I. Observation Space (State)
The agent receives a normalized continuous vector of 6 coordinates (`Box(low=0.0, high=9.0, shape=(6,))`):

$$\mathbf{S} = [\text{Player}_X, \; \text{Player}_Y, \; \text{Enemy}_X, \; \text{Enemy}_Y, \; \text{Target}_X, \; \text{Target}_Y]$$

- **Player Vector:** Coordinates $[P_X, P_Y]$ on a $10 \times 10$ matrix ($X \in [0, 9], Y \in [0, 9]$).
- **Enemy Vector:** Coordinates $[E_X, E_Y]$ representing the pursuer (Blinky).
- **Target Vector:** Coordinates $[T_X, T_Y]$ representing the collectible Power Pellet.

### II. Action Space
Discrete motor interface with 5 directional actions (`Discrete(5)`):
- `0`: Idle / Stay (Hold position)
- `1`: Move Left ($X - 1$)
- `2`: Move Right ($X + 1$)
- `3`: Move Up ($Y + 1$)
- `4`: Move Down ($Y - 1$)

### III. Mathematical Reward Function
The scalar feedback function is shaped to optimize pathfinding efficiency while severely penalizing collision:

| Trigger Event | Reward Scalar | Outcome | Purpose |
| :--- | :---: | :---: | :--- |
| **Step Traversal** | `-0.2` | Continue | Penalizes stalling and enforces the fastest path |
| **Progress Closer to Target** | `+0.5` | Continue | Reward shaping towards the objective |
| **Moving Away from Target** | `-0.5` | Continue | Discourages backtracking |
| **Wall Collision** | `-1.0` | Continue (Blocked) | Teaches boundary respect |
| **Proximity to Enemy (1 tile)** | `-3.0` | Continue | Threat bubble avoidance |
| **Target Acquisition** | `+100.0` | **Terminate** | Goal solved |
| **Enemy Collision** | `-100.0` | **Terminate** | Catastrophic failure |
| **Horizon Cutoff (60 Steps)** | `0.0` | **Truncate** | Stops non-converging loops |

---

## 🎮 2. Interactive Web UI (`index.html`)

A complete zero-dependency single-file application featuring:
- **Classic CRT Arcade Aesthetic**: Neon blue maze walls, pitch-black corridors, pellet dots, CRT scanlines, and screen flicker.
- **Animated Entities**: Chomping Pac-Man player facing the direction of travel, Blinky ghost with tracking eyeballs, and pulsating power pellet.
- **Synthesized 8-bit Audio**: Native browser **Web Audio API** oscillator synthesizing retro *waka-waka*, fruit-eating, siren alerts, and death sounds.
- **Dynamic Procedural Maze Generator**: Re-generates wall configurations on every successful run (`LAYOUT #1`, `LAYOUT #2`, ...).
- **Autonomous AI Mode**: Real-time **Breadth-First Search (BFS)** algorithm calculating the shortest obstacle-free path while executing dynamic evasion whenever Blinky threatens interception.
- **Human Demonstration Recorder**: Saves your gameplay transitions directly to `localStorage` and provides a one-click export to `human_play_data.json`.

---

## 🚀 3. Installation & Quickstart

### Prerequisites
- Python 3.8 to 3.11
- A browser (Chrome, Edge, Firefox, or Safari)

### Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/prathameshrodi/aigaming.git
cd aigaming

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install required dependencies
pip install gymnasium stable-baselines3 torch numpy
```

---

## 🧠 4. Training Pipelines

### Mode A: Standard Reinforcement Learning (From Scratch)
Train a PPO model directly inside the Gymnasium environment:
```bash
python train.py --timesteps 60000 --output pac2d_brain.zip
```

### Mode B: Training from Your Own Gameplay (Behavioral Cloning + RL)
1. Double-click [index.html](index.html) to open the game in your browser.
2. Play several games in **Manual Mode** using arrow keys to collect the target.
3. Click the green button: **`💾 DOWNLOAD DATA FOR AI`** to save `human_play_data.json`.
4. Move `human_play_data.json` into the repository directory.
5. Run the imitation training script:
```bash
python train_from_human.py --data human_play_data.json --epochs 25 --rl-steps 20000 --output pac2d_human_brain.zip
```

---

## 📺 5. Model Evaluation (Terminal Visualizer)

Evaluate your trained neural policy deterministically:
```bash
python enjoy.py --model pac2d_brain.zip --episodes 5 --delay 300
```
This streams an ASCII visualization of Pac-Man finding the fastest path and evading Blinky in real time directly inside your terminal console:

```text
--- [2D PAC-NAV] Step: 12/60 ---
    +------------------------------+
 9  | .  .  .  .  .  .  .  .  .  . |
 8  | .  #  #  .  #  #  .  #  #  . |
 7  | .  .  .  .  .  .  .  .  .  T |
 6  | .  #  .  #  #  #  #  .  #  . |
 5  | .  #  .  .  .  .  P  .  #  . |
 4  | .  #  .  #  #  #  #  .  #  . |
 3  | .  .  .  .  .  .  .  .  .  . |
 2  | .  #  #  .  #  #  .  #  #  . |
 1  | .  .  E  .  .  .  .  .  .  . |
 0  | .  .  .  .  .  .  .  .  .  . |
    +------------------------------+
        0  1  2  3  4  5  6  7  8  9 
Player: [6, 5] | Enemy: [2, 1] | Target: [9, 7]
```

---

## 📜 License
MIT License. Open for reinforcement learning research and game development experimentation.
