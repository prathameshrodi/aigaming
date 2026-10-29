import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from environment import Pac2DEnv

def train(total_timesteps=60000, model_path="pac2d_brain.zip"):
    print("[1] Initializing 2D Maze Navigation Environment...")
    env = Pac2DEnv()

    # Validate environment compliance
    check_env(env, warn=True)
    print("✓ Environment check passed!")

    print(f"[2] Configuring PPO Policy (Total timesteps: {total_timesteps})...")
    # PPO tuned for 2D maze pathfinding with obstacle and enemy evasion
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
    )

    print("[3] Training PPO Agent on Fastest Path & Evasion...")
    model.learn(total_timesteps=total_timesteps)

    print(f"[4] Storing Network Weights -> {model_path}...")
    model.save(model_path.replace(".zip", ""))
    print(f"✓ Training complete! Weights successfully saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent on 2D Pac-Man Navigation.")
    parser.add_argument("--timesteps", type=int, default=60000, help="Total training timesteps (default: 60000)")
    parser.add_argument("--output", type=str, default="pac2d_brain.zip", help="Output model weights file path")
    args = parser.parse_args()

    train(total_timesteps=args.timesteps, model_path=args.output)
