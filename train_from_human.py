import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from stable_baselines3 import PPO
from environment import Pac2DEnv

class HumanDemonstrationDataset(Dataset):
    """
    Loads recorded state-action pairs from human demonstration JSON files.
    Input: Observation vector of length 6 [P_x, P_y, E_x, E_y, T_x, T_y]
    Target: Action integer (0..4)
    """
    def __init__(self, data_path="human_play_data.json"):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Demonstration file '{data_path}' not found! Play some games in the browser and export your data.")

        with open(data_path, "r") as f:
            raw_data = json.load(f)

        observations = []
        actions = []

        # Parse recorded episodes
        for ep in raw_data:
            steps = ep.get("steps", [])
            outcome = ep.get("outcome", "")

            # Filter for successful episodes or all steps where player didn't hit a wall/ghost
            for step in steps:
                obs = step["obs"]       # [px, py, ex, ey, tx, ty]
                act = step["action"]    # 0..4
                observations.append(obs)
                actions.append(act)

        print(f"Loaded {len(observations)} state-action transitions from {len(raw_data)} recorded human episodes.")
        self.observations = torch.tensor(np.array(observations, dtype=np.float32))
        self.actions = torch.tensor(np.array(actions, dtype=np.int64))

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


def train_from_human_demonstrations(
    data_path="human_play_data.json",
    output_model="pac2d_human_brain.zip",
    pretrain_epochs=25,
    rl_finetune_timesteps=20000
):
    print("=======================================================")
    print(" PHASE 1: BEHAVIORAL CLONING (IMITATION LEARNING)")
    print("=======================================================")

    dataset = HumanDemonstrationDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize environment and PPO model
    env = Pac2DEnv()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        verbose=1
    )

    # Extract the Actor/Policy network from PPO's policy
    policy_net = model.policy
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Pretraining PPO policy on human play data for {pretrain_epochs} epochs...")
    policy_net.train()

    for epoch in range(1, pretrain_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_obs, batch_actions in dataloader:
            optimizer.zero_grad()

            # PPO policy network evaluates distribution over action space
            distribution = policy_net.get_distribution(batch_obs)
            logits = distribution.distribution.logits

            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_actions)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_actions).sum().item()
            total += len(batch_actions)

        avg_loss = total_loss / total
        acc = (correct / total) * 100.0
        if epoch % 5 == 0 or epoch == pretrain_epochs:
            print(f"Epoch [{epoch:02d}/{pretrain_epochs:02d}] | CrossEntropy Loss: {avg_loss:.4f} | Human Action Match: {acc:.1f}%")

    print("\n✓ Phase 1 complete: Policy initialized with human decision patterns!")

    if rl_finetune_timesteps > 0:
        print("\n=======================================================")
        print(" PHASE 2: REINFORCEMENT LEARNING FINE-TUNING")
        print("=======================================================")
        print(f"Refining pretrained human policy with {rl_finetune_timesteps} RL timesteps...")
        model.learn(total_timesteps=rl_finetune_timesteps)

    # Save finalized model
    save_path = output_model.replace(".zip", "")
    model.save(save_path)
    print(f"\n🎉 Successfully saved combined human+RL model to '{output_model}'!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model from recorded human gameplay.")
    parser.add_argument("--data", type=str, default="human_play_data.json", help="Path to exported human play JSON")
    parser.add_argument("--epochs", type=int, default=25, help="Imitation learning epochs (default: 25)")
    parser.add_argument("--rl-steps", type=int, default=20000, help="RL fine-tuning timesteps (default: 20000)")
    parser.add_argument("--output", type=str, default="pac2d_human_brain.zip", help="Output model zip path")
    args = parser.parse_args()

    train_from_human_demonstrations(
        data_path=args.data,
        output_model=args.output,
        pretrain_epochs=args.epochs,
        rl_finetune_timesteps=args.rl_steps
    )
