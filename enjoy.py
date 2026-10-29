import time
import argparse
import os
from stable_baselines3 import PPO
from environment import Pac2DEnv

def enjoy(model_path="pac2d_brain.zip", episodes=5, delay_ms=300):
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            raise FileNotFoundError(f"Model '{model_path}' not found. Please run train.py first.")

    print(f"Loading trained 2D agent from '{model_path}'...")
    env = Pac2DEnv(render_mode="human")
    model = PPO.load(model_path)

    action_names = {0: "Idle", 1: "Move Left", 2: "Move Right", 3: "Move Up", 4: "Move Down"}

    for ep in range(1, episodes + 1):
        print(f"\n==========================================")
        print(f"        STARTING 2D EPISODE {ep} / {episodes}")
        print(f"==========================================")
        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0.0

        time.sleep(delay_ms / 1000.0)

        while not done:
            step += 1
            action, _states = model.predict(obs, deterministic=True)
            action_int = int(action)

            obs, reward, terminated, truncated, info = env.step(action_int)
            total_reward += reward
            done = terminated or truncated

            print(f"Step {step} | Action: [{action_int}] {action_names.get(action_int, 'Unknown')} | Step Reward: {reward:.1f} | Cumulative: {total_reward:.1f}")

            if terminated:
                if reward >= 90.0:
                    print(f"\n🎉 SUCCESS! Target Acquired in {step} steps! Fastest Path Verified. Final Reward: {total_reward:.1f}")
                else:
                    print(f"\n💥 CAUGHT! Intercepted by Enemy at step {step}! Final Reward: {total_reward:.1f}")
            elif truncated:
                print(f"\n⏱️ HORIZON REACHED: Maximum steps exceeded. Final Reward: {total_reward:.1f}")

            time.sleep(delay_ms / 1000.0)

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 2D agent navigation.")
    parser.add_argument("--model", type=str, default="pac2d_brain.zip", help="Path to model zip file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes (default: 5)")
    parser.add_argument("--delay", type=int, default=300, help="Delay in ms (default: 300)")
    args = parser.parse_args()

    enjoy(model_path=args.model, episodes=args.episodes, delay_ms=args.delay)
