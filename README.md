# ran-slicing-qtable

A lightweight tabular Q-learning baseline for RAN (Radio Access Network) network slicing resource allocation.

---

## Project Goal

This repository provides a **skeleton** for a small, discretized tabular Q-learning agent that interacts with the [network-slicing Gym environment](https://github.com/jjalcaraz-upct/network-slicing). It is **not** a fork of the original repository and does **not** reimplement the environment.

The goal is to establish a clean, readable Q-table baseline as a starting point for studying reinforcement learning applied to RAN resource allocation—without any deep learning frameworks.

---

## Why Network Slicing is Relevant to Wireless RL

Modern mobile networks (4G/5G) serve multiple logical "slices" simultaneously—each with different QoS/SLA requirements (e.g., eMBB, URLLC, mMTC). Allocating time-frequency resource blocks (RBs) efficiently across slices is a sequential decision-making problem well-suited to RL:

- The state captures current load, queue sizes, and SLA status per slice.
- The action is the RB budget assigned to each slice for the next observation period.
- The reward encodes SLA fulfillment and resource efficiency.

---

## Why Raw Tabular Q-Learning May Not Work Directly

The original environment has a potentially large or partially continuous observation space (load levels, queue sizes, channel quality, etc.) and a large combinatorial action space (all possible RB distributions across N slices). These make a direct lookup table infeasible:

- A table entry is required for every `(state, action)` pair.
- Continuous or high-dimensional states yield an exponential number of bins.
- The original paper uses Stable-Baselines v2 PPO and Keras-RL DQN, not tabular methods.

---

## Why a Small Discretized Version is Needed

To make tabular Q-learning tractable, this skeleton introduces:

1. **StateDiscretizer** — maps the raw observation vector into a small tuple of discrete bins.
2. **ActionMapper** — maps a small set of hand-crafted RB-allocation strategies to environment actions, replacing the full combinatorial action space.
3. **RewardAdapter** — normalizes the raw environment reward/info into a scalar suitable for Q-updates.
4. **QTableAgent** — stores Q-values in a Python dict keyed by discrete `(state, action)` tuples.

---

## ⚠️ Important Warning

> **This project does NOT include the network-slicing environment.**
>
> The original Gym environment must be installed separately from:
> https://github.com/jjalcaraz-upct/network-slicing
>
> This repository is **only a lightweight Q-table baseline wrapper**.
> It is not runnable until the original environment is installed and the correct `env_id` / action format are verified by inspecting the original repo.

---

## Expected Setup Flow

1. Clone and install **this** repository (the Q-table baseline).
2. Separately clone and install the **original network-slicing environment** (see `scripts/setup_network_slicing_env.sh`).
3. Verify the correct `env_id` string and observation/action format from the original repo.
4. Update `configs/qlearning.yaml` with the correct `env_id`.
5. Run training.

---

## Installation (this project)

Requires Python ≥ 3.10. Uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv if not already available
pip install uv

# Create a virtual environment and install this project
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

Or with plain pip:

```bash
pip install -e .
```

---

## Setup the Original Network-Slicing Environment

```bash
bash scripts/setup_network_slicing_env.sh
```

See the script for step-by-step instructions. The original repo uses legacy dependencies (`gym==0.15.3`, `numpy==1.19.1`, `pandas==0.25.2`), so a separate virtual environment is recommended.

---

## Running Training

```bash
bash scripts/run_train.sh
```

Or directly:

```bash
python -m ran_slicing_qtable.train --config configs/qlearning.yaml
```

---

## Running Evaluation

```bash
python -m ran_slicing_qtable.evaluate --config configs/qlearning.yaml
```

---

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── qlearning.yaml          # Hyperparameters and env config
├── scripts/
│   ├── setup_network_slicing_env.sh   # Setup the original env
│   └── run_train.sh            # Launch training
├── src/
│   └── ran_slicing_qtable/
│       ├── __init__.py
│       ├── q_table_agent.py    # Tabular Q-learning agent
│       ├── state_discretizer.py # Continuous → discrete state mapping
│       ├── action_mapper.py    # Small action set → env RB allocation
│       ├── reward_adapter.py   # Raw reward/info → scalar reward
│       ├── env_factory.py      # Gym env creation with error handling
│       ├── train.py            # Training loop
│       ├── evaluate.py         # Greedy evaluation loop
│       └── utils.py            # Shared utilities
└── notes/
    └── network_slicing_qtable.md  # Design notes
```

---

## Notes

See [`notes/network_slicing_qtable.md`](notes/network_slicing_qtable.md) for design rationale, known challenges, and suggested next steps.
