# Two-Stage PPO with Opponent Modeling for Leduc Hold’em

This repository implements a **two-stage Proximal Policy Optimization (PPO)** framework for **Leduc Hold’em**, using the PettingZoo environment.
In **Stage 2**, the agent is augmented with a **learned opponent model** that predicts the opponent’s next action and incorporates this prediction into the policy input.

The project is designed for **research and educational purposes**, focusing on:

* Multi-agent reinforcement learning
* Self-play and frozen-opponent evaluation
* Opponent modeling as auxiliary supervised learning
* PPO with TD(λ) / GAE value estimation

---

## Project Structure

```
.
├── main_PPO_v2.py                 # Stage 1: PPO training vs random opponent
├── main_PPO_v2_selfplay.py        # Stage 2: PPO training vs frozen PPO opponent
├── PPO_v2.py                      # PPO agent, policy/value networks, training loop
├── opponent_model.py              # Supervised opponent action predictor
├── utils.py                       # Plotting & smoothing utilities
├── models/
│   └── stage1_ppo_random_*.pth    # Saved Stage 1 PPO models
├── results/
│   ├── *.png                      # Training curves and opponent model metrics
└── README.md
```

---

## Environment

* **Game**: Leduc Hold’em
* **Framework**: [PettingZoo](https://pettingzoo.farama.org/) (`leduc_holdem_v4`)
* **Observation space**: 36-dim vector
* **Action space**:

  ```
  0: Call
  1: Raise
  2: Fold
  3: Check
  ```

---

## Core Ideas

### Stage 1 — Baseline PPO

* Train a PPO agent against a **random opponent**
* Learn a stable baseline policy
* Save the trained policy for later evaluation

### Stage 2 — PPO with Opponent Modeling

* Freeze the Stage-1 PPO agent as the **opponent**
* Train a new PPO agent that:

  * Learns a **supervised opponent model** to predict opponent actions
  * Concatenates predicted opponent action probabilities to the observation
  * Uses this augmented state for policy optimization

---

## Algorithm Overview

### PPO Agent

* **Policy network**: 2-layer MLP
* **Value network**: 1-hidden-layer MLP
* **Optimization**:

  * PPO clipped objective
  * TD(λ) or Monte Carlo value targets
  * GAE-style advantage estimation
* **Action masking** to ensure legality

### Opponent Model

* Supervised multi-class classifier
* Input: opponent observation
* Output: action probability distribution (softmax)
* Loss: cross-entropy
* Trained online during gameplay

---

## Installation

```bash
pip install torch numpy matplotlib pettingzoo
```

Make sure you have a working CUDA setup if you want GPU acceleration.

---

## Running Experiments

### Stage 1 — Train PPO vs Random Opponent

```bash
python main_PPO_v2.py
```

This will:

* Train PPO against a random opponent
* Save the trained policy to `models/`
* Plot reward and win rate curves

---

### Stage 2 — Train PPO with Opponent Model vs Frozen PPO Opponent

```bash
python main_PPO_v2_selfplay.py
```

This will:

1. Run PPO **without** opponent modeling
2. Run PPO **with** opponent modeling
3. Compare:

   * Average reward
   * Win rate
4. Plot:

   * Training curves
   * Opponent model loss & accuracy

---

## Key Hyperparameters

```python
num_episodes     = 1000
policy_iter      = 30
policy_epochs    = 20
value_epochs     = 10
policy_batch_size= 32
value_batch_size = 16
gamma            = 0.99
lambda (TD/GAE)  = 0.9
clip_eps         = 0.2
```

All hyperparameters are easily adjustable in the `main_*.py` files.

---

## Outputs & Metrics

* **Reward curves** (moving average)
* **Win rate** per training iteration
* **Opponent model accuracy**
* **Opponent model loss**

Plots are saved under `results/`.

---

## Acknowledgements

* PettingZoo team for the Leduc Hold’em environment
* OpenAI PPO formulation
* Farama Foundation ecosystem

---

## License

This project is provided for **research and educational use**.
Add a license file if you plan to redistribute or publish.

---

First, modify the directory path of MODEL_PATH and image save path to your own.

Stage 1: Run main_PPO_v2.py  trained model saved to the 'models' folder, result plot saved to the 'results' folder

Stage 2: Run main_PPO_v2_selfplay.py  load model from the 'models' folder, result plot saved to the 'results' folder

---

You can achieve a similar task using the files ending with policy_gradient, this is different in that is uses a naive policy-gradient algorithm instead of PPO.
