# utils.py
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import os

def plot_results(scores_no_model, scores_with_model, figname, win_rate, win_rate_model):
    plt.figure(figsize=(12, 6))
    
    plt.plot(scores_no_model, label="Without Opponent Model", linewidth=2, color='blue', alpha=0.8)
    if scores_with_model is not None:
        plt.plot(scores_with_model, label="With Opponent Model", linewidth=2, color='red', alpha=0.8)
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Reward (200-ep MA)", fontsize=12)
    
    info_text = f"No Model Win Rate: {win_rate:.3f}"
    if scores_with_model is not None:
        info_text += f"\nWith Model Win Rate: {win_rate_model:.3f}"
    
    plt.text(
        0.98, 0.98,
        info_text,
        transform=plt.gca().transAxes,
        ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
        linespacing=1.5
    )
    
    plt.legend(loc='lower right', fontsize=11)
    plt.title(f"{figname} - Training Performance", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_dir = r"D:\lkh\康奈尔\25 Fall\special topics in gen AI\poker_code\code\results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(os.path.join(save_dir, f"{figname}.png"), dpi=300, bbox_inches='tight')
    plt.show()

def moving_average(x, window=100):
    x = np.array(x, dtype=float)
    ma = []

    for i in range(len(x)):
        if i < window:
            # average all data so far
            ma.append(x[:i+1].mean())
        else:
            # sliding window average
            ma.append(x[i-window+1:i+1].mean())

    return np.array(ma)[50:]


# def moving_average(x, window=100): 
#     return np.convolve(x, np.ones(window)/window, mode="valid")


def plot_training_curve(iter_rewards, title="PPO Training Curve", ylabel="Avg Reward"):
    iterations = list(range(1, len(iter_rewards) + 1))

    plt.figure(figsize=(8,5))
    plt.plot(iterations, iter_rewards, marker='o', linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel("Policy Iteration", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


