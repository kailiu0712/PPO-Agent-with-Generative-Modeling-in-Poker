# main_PPO_v2_selfplay.py
import numpy as np
import torch
from pettingzoo.classic import leduc_holdem_v4
from PPO_v2 import RLAgent
from opponent_model import OpponentPredictor
from utils import plot_results, moving_average
from statistics import mean
import matplotlib.pyplot as plt
import os
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Leduc official specs:
# Observation shape = (36,)
# Actions = {0: Call, 1: Raise, 2: Fold, 3: Check}
ACTION_SPACE = 4 # 0-call 1-raise 2-fold 3-check (raise amount is fixed)
HIDDEN = 32
MODEL_DIR = r"D:\lkh\康奈尔\25 Fall\special topics in gen AI\poker_code\code\2-stage_ppo_v2\models"  
MODEL_NAME = "stage1_ppo_random_1212_2.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

def run_experiment(use_opponent_model=False, num_episodes=100, value_batch_size=64, policy_batch_size=100, policy_iter=10, policy_epochs=10, value_epochs=10, method='TD', lam=0.9):
    ## T is the episodes number we collect for PPO update per time.

    env = leduc_holdem_v4.env()
    env.reset(seed=42)

    # ---- detect true observation dimension (avoid mismatches) ----
    first_obs, _, _, _, _ = env.last()
    TRUE_OBS_DIM = len(first_obs["observation"])

    print(f"Detected observation dim = {TRUE_OBS_DIM}")

    # ---- opponent model only if enabled ----
    opp_model = (
        OpponentPredictor(TRUE_OBS_DIM, ACTION_SPACE)
        if use_opponent_model
        else None
    )

    # FROZEN opponent
    opp_agent = RLAgent(TRUE_OBS_DIM, ACTION_SPACE, use_opponent_model=False)
    opp_agent.policy.load_state_dict(torch.load(MODEL_PATH))
    opp_agent.policy.eval()
    print("[Phase 2] Loaded frozen opponent agent.")

    # RL agent input = either obs or obs+predicted_actions
    rl_input_dim = TRUE_OBS_DIM + (ACTION_SPACE if use_opponent_model else 0)
    rl_agent = RLAgent(
        rl_input_dim,
        ACTION_SPACE,
        use_opponent_model=use_opponent_model,
        opponent_model=opp_model,
        clip_eps=0.2
    )

    reward_hist, best_rewards_hist, win_rate = rl_agent.ppo_training(env, num_episodes=num_episodes, value_batch_size=value_batch_size, policy_batch_size=policy_batch_size,
                                                           policy_iter=policy_iter, policy_epochs=policy_epochs,
                          value_epochs=value_epochs, use_opponent_model=use_opponent_model, opp_model=opp_model, method=method, lam=lam, opp_agent=opp_agent)

    return reward_hist, best_rewards_hist, win_rate, rl_agent

def main():
    '''
    ---- Params ----
    use_opponent_model: (True/False) whether to use opponent model
    num_episodes: number of episodes for simulation
    value_batch_size: batch size for value iteration
    policy_batch_size: batch size for policy iteration
    policy_iter: total policy iterations
    policy_epochs: epochs to train policy network
    value_epochs: epochs to train value network
    method: method for value updates (TD(lamda) or Monte Carlo)
    lam: lambda for TD(lambda)
    figname: name of the plot
    '''
    use_opponent_model = False
    num_episodes = 1000
    value_batch_size = 16
    policy_batch_size = 32
    policy_iter = 30
    policy_epochs = 20
    value_epochs = 10
    method = "TD"
    lam = 0.9
    figname = 'stage_1_PPO_v2_stage2_1212_explore'

    start_time = time.time()

    reward_hist, best_rewards_hist, win_rate, agent = run_experiment(use_opponent_model=use_opponent_model, num_episodes=num_episodes,
                                                              value_batch_size=value_batch_size, policy_batch_size=policy_batch_size,
                                                              policy_iter=policy_iter, policy_epochs=policy_epochs, value_epochs=value_epochs,
                                                              method=method, lam=lam)
    
    # Use opponent predictor this time
    use_opponent_model = True

    reward_hist_opp, best_rewards_hist, win_rate_opp, agent = run_experiment(use_opponent_model=use_opponent_model, num_episodes=num_episodes,
                                                              value_batch_size=value_batch_size, policy_batch_size=policy_batch_size,
                                                              policy_iter=policy_iter, policy_epochs=policy_epochs, value_epochs=value_epochs,
                                                              method=method, lam=lam)
    
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

    print(f'Final win rate (without opponent model): {win_rate[-1]}')
    print(f'Final win rate (with opponent model): {win_rate_opp[-1]}')
    print(f'Average win rate (without opponent model): {np.mean(np.array(win_rate))}')
    print(f'Average win rate (with opponent model): {np.mean(np.array(win_rate_opp))}')
    print(f'Average win rate diff: {np.mean(np.array(win_rate_opp) - np.array(win_rate))}')

    iterations = range(len(reward_hist))  
    win_iterations = range(len(win_rate)) 

    # Plot results
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Color settings
    reward_color_wo = "steelblue"
    reward_color_w  = "skyblue"
    win_color_wo = "steelblue"
    win_color_w  = "skyblue"

    # ---- Plot reward ----
    ax1.plot(iterations, reward_hist, label="Reward (no opponent model)", linewidth=2, color=reward_color_wo)
    ax1.plot(iterations, reward_hist_opp, label="Reward (with opponent model)", linewidth=2, color=reward_color_w)
    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)

    # ---- Plot win rate ----
    ax2 = ax1.twinx()
    ax2.scatter(win_iterations, win_rate, label="Win Rate (no opponent model)", color=win_color_wo, s=30, marker="o")
    ax2.scatter(win_iterations, win_rate_opp, label="Win Rate (with opponent model)", color=win_color_w, s=30, marker="o")
    ax2.set_ylabel("Win Rate")

    # ---- Combine legends ----
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower right")

    plt.title("Reward and Win Rate during PPO Training (stage 2)")
    plt.tight_layout()
    plt.savefig(rf'D:\lkh\康奈尔\25 Fall\special topics in gen AI\poker_code\code\2-stage_ppo_v2\results\{figname}.png')
    plt.show()

    # Plot Opponent Model Training Metrics
    if use_opponent_model:
        opp_model_losses = agent.opp_model_losses
        opp_model_accs = [c for c in agent.opp_model_corrects]
        steps = list(range(len(opp_model_losses)))

        loss_avg = [np.mean(opp_model_losses[max(0, i-500):i+1]) for i in range(len(opp_model_losses))]
        acc_avg = [np.mean(opp_model_accs[max(0, i-500):i+1]) for i in range(len(opp_model_accs))]

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(steps, loss_avg, label="Train Loss", color="crimson", linewidth=1.5)
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss", color="crimson")
        ax1.tick_params(axis='y', labelcolor="crimson")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(steps, acc_avg, label="Accuracy", color="blue", linewidth=1.5)
        ax2.set_ylabel("Accuracy", color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", bbox_to_anchor=(0.75, 0.5))

        plt.title("Opponent Model Training Loss & Accuracy")
        plt.tight_layout()
        plt.savefig(rf'D:\lkh\康奈尔\25 Fall\special topics in gen AI\poker_code\code\2-stage_ppo_v2\results\opponent_model_metrics_{figname}.png')

if __name__ == "__main__":
    main()