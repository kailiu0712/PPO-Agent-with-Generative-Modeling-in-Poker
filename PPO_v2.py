# PPO_v2.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.physics.quantum.density import entropy
from torch.distributions import Categorical

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.policy_head(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.value_head(x)

class RLAgent:
    def __init__(self, obs_dim, act_dim, lr=1e-3, gamma=0.99, lam=0.9, clip_eps=0.1, use_opponent_model=False, opponent_model=None):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.use_opponent_model = use_opponent_model
        self.opponent_model = opponent_model
        self.policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
        self.value = ValueNetwork(obs_dim).to(DEVICE)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr * 0.2)
        self.opp_model_losses = []
        self.opp_model_corrects = []
        self.opp_model_total = 0


    def simulate_episodes(self, env=None, num_episodes=100, use_opponent_model=False, opp_model=None, opp_agent=None):
        """
        Simulate num_episodes game
        Return: flatten episodes tensors
        """
        states_flatten, actions_flatten = [], []
        rewards_flatten, dones_flatten = [], []
        avg_reward = 0.0
        episode = 0
        win_count = 0
        tie_count = 0

        while episode < num_episodes:
            env.reset()
            states, actions = [], []
            rewards, dones = [], []

            for name in env.agent_iter():

                obs, reward, termination, truncation, info = env.last()

                if name == "player_0":
                    rewards.append(reward)
                    dones.append(torch.tensor(float(termination or truncation)))

                if termination or truncation:
                    env.step(None)
                    continue

                mask = obs["action_mask"]
                state = obs["observation"]

                if name == "player_0":
                    if use_opponent_model and opp_model:
                        opp_pred = opp_model.predict(state) 
                        state = np.concatenate([state, opp_pred]) # We concatenate the predicted actions to original states

                    else:
                        # do NOT wrap in concatenate → just use raw state
                        pass
                    states.append(state)

                    action = self.get_action(state, mask)
                    actions.append(action)

                else:  # opponent moves
                    if opp_agent:
                        action = opp_agent.get_action(state, mask) # stage 2: opponent is a trained agent
                    else:
                        valid = [i for i, m in enumerate(mask) if m == 1]
                        action = np.random.choice(valid) if valid else 0 # stage 1: opponent plays a random strategy

                    # train opponent model via supervised target
                    if use_opponent_model and opp_model:
                        loss, acc = opp_model.train_step(state, action)
                        self.opp_model_losses.append(loss)
                        self.opp_model_corrects.append(acc)
                        self.opp_model_total += 1

                env.step(action)
            env.close()

            if len(rewards) == 1:
                continue
            episode += 1
            if rewards[-1] > 0:
                win_count += 1
            elif rewards[-1] == 0:
                tie_count += 1

            rewards = rewards[1:]  # Drop the first reward
            dones = dones[1:]  # Also drop the first game indicator
            avg_reward += (rewards[-1] * (self.gamma ** (len(rewards) - 1)) - avg_reward) / episode
            rewards_flatten += rewards
            states_flatten += states
            actions_flatten += actions
            dones_flatten += dones

        states_flatten = np.array(states_flatten, dtype=np.float32)
        states_tensor = torch.from_numpy(states_flatten)
        actions_tensor = torch.tensor(actions_flatten, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards_flatten, dtype=torch.float32)
        dones_tensor = torch.tensor(dones_flatten, dtype=torch.float32)
        win_rate = (win_count + 0.5 * tie_count) / num_episodes

        return states_tensor, actions_tensor, rewards_tensor, dones_tensor, avg_reward, win_rate

    def get_action(self, obs, mask):
        obs = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = self.policy(obs)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=DEVICE)
        # Mask logits BEFORE softmax
        masked_logits = logits.masked_fill(~mask_tensor, -1e9)
        probs = torch.softmax(masked_logits, dim=-1).squeeze()
        dist = Categorical(probs)
        action = dist.sample()
        return action.item()

    def compute_td_lambda_targets(self, rewards, values, dones, lam=0.9, methods="TD"):
        """
        All flatten tensors
        rewards: (T,)
        dones: (T,), True when episodes end
        Return: TD(lam) target to update value net
        """
        T = len(rewards)
        td_lambda_targets = torch.zeros_like(rewards, device=DEVICE)
        if methods == "TD":
            cumulative_td_error = 0

            for t in reversed(range(T)):
                if t == T - 1:
                    td_error = rewards[t] - values[t]
                else:
                    td_error = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t] 
                cumulative_td_error = td_error + self.gamma * lam * cumulative_td_error * (1 - dones[t]) # TD lambda
                td_lambda_targets[t] = values[t] + cumulative_td_error

        else: 
            for t in reversed(range(T)):
                if t == T - 1:
                    td_lambda_targets[t] = rewards[t]
                else:
                    td_lambda_targets[t] = rewards[t] + self.gamma * td_lambda_targets[t+1] * (1 - dones[t]) # Monte Carlo

        return td_lambda_targets

    def compute_td_adv(self, rewards, values, dones, lam=0):
        '''
        Apply a GAE approach to compute the advantage functions
        '''
        T = len(rewards)
        advantages = torch.zeros_like(rewards, device=DEVICE)
        gae = 0 

        for t in reversed(range(T)):
            if t == T - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * lam * gae * (1 - dones[t])
            advantages[t] = gae

        return advantages

    def value_training(self, states, rewards, dones, value_epoch=10, batch_size=10, method='TD'):
        for value_epoch in range(value_epoch):
            with torch.no_grad():
                values = self.value(states).squeeze(1)

            td_targets = self.compute_td_lambda_targets(rewards, values, dones, methods=method)

            perm = torch.randperm(states.shape[0])
            for idx in range(0, states.shape[0], batch_size):
                batch_idx = perm[idx: idx + batch_size]
                states_batch = states[batch_idx]
                returns_batch = td_targets[batch_idx]
                value_pred = self.value(states_batch).squeeze(1)
                value_loss = nn.MSELoss()(value_pred, returns_batch)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def ppo_training(self, env, num_episodes=100, value_batch_size=64, policy_batch_size=50, policy_iter=10, policy_epochs=10, value_epochs=10, use_opponent_model=False, opp_model=None, method='TD', lam=0.9, opp_agent=None):
        ## All flatten vectors
        best_avg_reward = -np.inf
        reward_hist = []
        best_reward_hist = []
        win_rate = []

        for policy_itr in range(policy_iter):
            print(f"Training iteration: {policy_itr+1}/{policy_iter}:")
            states_flatten, actions_flatten, rewards_flatten, dones_flatten, curr_avg_reward, win_rate_itr = self.simulate_episodes(
                env=env, num_episodes=num_episodes, use_opponent_model=use_opponent_model, opp_model=opp_model, opp_agent=opp_agent)

            win_rate.append(win_rate_itr)

            if curr_avg_reward > best_avg_reward:
                best_avg_reward = curr_avg_reward
            reward_hist.append(curr_avg_reward)
            best_reward_hist.append(best_avg_reward)

            self.value_training(states_flatten, rewards_flatten, dones_flatten, value_epoch=value_epochs, batch_size=value_batch_size, method=method)

            with torch.no_grad():
                logits = self.policy(states_flatten)
            dist = Categorical(logits=logits)
            old_log_probs = dist.log_prob(actions_flatten)
            # Compute advantages using value net (one-step TD)
            with torch.no_grad():
                values = self.value(states_flatten).squeeze(1)
            advantages = self.compute_td_adv(rewards_flatten, values, dones_flatten, lam=lam)

            for _ in range(policy_epochs):
                perm = torch.randperm(states_flatten.shape[0]) # Use Random reshuffling as scan order
                for idx in range(0, states_flatten.shape[0], policy_batch_size):
                    batch_idx = perm[idx: idx + policy_batch_size] 

                    s = states_flatten[batch_idx]
                    a = actions_flatten[batch_idx]
                    old_lp = old_log_probs[batch_idx]
                    adv = advantages[batch_idx]  # (B, )

                    # Policy loss
                    logits = self.policy(s)  # (B, S)
                    dist = Categorical(logits=logits)
                    log_probs = dist.log_prob(a)  # (B, )
                    ratio = torch.exp(log_probs - old_lp)

                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    policy_loss = -torch.min(ratio * adv, clipped_ratio * adv).mean() # PPO objective

                    # Optimize
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    self.policy_optimizer.step()

        return reward_hist, best_reward_hist, win_rate