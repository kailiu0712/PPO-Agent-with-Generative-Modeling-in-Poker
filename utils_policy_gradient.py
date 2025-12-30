# utils.py
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

def plot_results(scores_no_model, scores_with_model, figname):
    plt.plot(scores_no_model, label="Without Opponent Model")
    if scores_with_model is not None:
        plt.plot(scores_with_model, label="With Opponent Model")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.title("Impact of Opponent Modeling on Leduc Hold'em")
    plt.savefig(rf'D:\lkh\康奈尔\25 Fall\special topics in gen AI\poker_code\cs224r_FP-my-feature-branch\cs224r_FP-my-feature-branch\project_policy_gradient\results\{figname}.png')
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

