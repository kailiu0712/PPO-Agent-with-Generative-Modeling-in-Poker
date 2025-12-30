# opponent_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OpponentModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


class OpponentPredictor:
    """Train a simple supervised model to predict opponent's next action."""
    def __init__(self, obs_dim, act_dim, lr=1e-3):
        self.model = OpponentModel(obs_dim, act_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, state, action):
        self.model.train()
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long, device=DEVICE)
        pred = self.model(state)
        loss = self.loss_fn(pred, action)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, state):
        """Return one-hot vector of predicted opponent action probabilities."""
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            return self.model(state).squeeze().cpu().numpy()
