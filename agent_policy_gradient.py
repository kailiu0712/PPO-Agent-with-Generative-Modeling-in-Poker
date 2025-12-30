# agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RLAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99, use_opponent_model=False, opponent_model=None):
        self.gamma = gamma
        self.use_opponent_model = use_opponent_model
        self.opponent_model = opponent_model
        self.policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def get_action(self, obs, mask):
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = self.policy(obs).squeeze()
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=DEVICE)
        # Mask logits BEFORE softmax
        masked_logits = logits.masked_fill(~mask_tensor, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)

        # === DEBUGGING: detect NaN ===
        if torch.isnan(probs).any():
            print("\n====== DEBUG INFO ======")
            print("obs:", obs)
            print("raw logits:", logits)
            print("mask:", mask)
            print("masked logits:", masked_logits)
            print("probs BEFORE softmax:", logits)
            print("probs AFTER softmax:", probs)
            print("========================\n")
            raise RuntimeError("NaN detected in probs — see debug info above.")

        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


    def update_policy(self, log_probs, rewards):
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = [ -lp * Gt for lp, Gt in zip(log_probs, returns) ]
        loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
