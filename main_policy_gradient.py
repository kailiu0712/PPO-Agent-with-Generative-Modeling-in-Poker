# main.py
import numpy as np
import torch
from pettingzoo.classic import leduc_holdem_v4
from agent_policy_gradient import RLAgent
from opponent_model_policy_gradient import OpponentPredictor
from utils_policy_gradient import plot_results, moving_average
from statistics import mean

EPISODES = 600

# Leduc official specs:
# Observation shape = (36,)
# Actions = {0: Call, 1: Raise, 2: Fold, 3: Check}
ACTION_SPACE = 4
HIDDEN = 32
MODEL_PATH = "trained_agent_random.pth"

def run_experiment(use_opponent_model=False):

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

    # RL agent input = either obs or obs+predicted_actions
    rl_input_dim = TRUE_OBS_DIM + (ACTION_SPACE if use_opponent_model else 0)
    rl_agent = RLAgent(
        rl_input_dim,
        ACTION_SPACE,
        use_opponent_model=use_opponent_model,
        opponent_model=opp_model,
    )

    all_rewards = []

    for ep in range(1, EPISODES + 1):
        env.reset()
        log_probs, rewards = [], []

        for name in env.agent_iter():

            obs, reward, termination, truncation, info = env.last()
            # if ep < 50:
                # print(name, obs, reward, termination, info)
            if name == "player_0":
                rewards.append(reward)  

            if termination or truncation:
                env.step(None)
                continue

            mask = obs["action_mask"]
            state = obs["observation"]

            if name == "player_0":
                # print('0', reward)
                # --- append opponent prediction ---
                if use_opponent_model and opp_model:
                    opp_pred = opp_model.predict(state)
                    state = np.concatenate([state, opp_pred])
                else:
                    # do NOT wrap in concatenate → just use raw state
                    pass

                action, log_prob = rl_agent.get_action(state, mask)
                log_probs.append(log_prob)                

            else:  # opponent moves
                valid = [i for i, m in enumerate(mask) if m == 1]
                action = np.random.choice(valid) if valid else 0

                # train opponent model via supervised target
                if use_opponent_model and opp_model:
                    opp_model.train_step(state, action)

            env.step(action)
        env.close()

        if log_probs and rewards:
            backprop_rewards = [rewards[-1] for _ in rewards]
            # update RL agent
            rl_agent.update_policy(log_probs, backprop_rewards)
            # rl_agent.update_policy(log_probs, rewards)

        all_rewards.append(sum(rewards))

        if ep % 50 == 0:
            avg = mean(all_rewards)
            print(f"[Episode {ep}/{EPISODES}] Avg Reward = {avg:.3f}")

    torch.save(rl_agent.policy.state_dict(), MODEL_PATH)
    print(f"[Phase 1] Saved trained model to {MODEL_PATH}")

    return moving_average(all_rewards, 200)


def main():

    figname = 'stage1_1'
    print("=== Training WITHOUT opponent model ===")
    scores_no_model = run_experiment(use_opponent_model=False)

    # print("\n=== Training WITH opponent model ===")
    # scores_with_model = run_experiment(use_opponent_model=True)

    plot_results(scores_no_model, None, figname)


if __name__ == "__main__":
    main()
