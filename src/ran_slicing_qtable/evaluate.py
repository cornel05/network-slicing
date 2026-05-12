"""
evaluate.py — Greedy evaluation of a trained Q-table policy.

Usage:
    python -m ran_slicing_qtable.evaluate --config configs/qlearning.yaml

Loads the saved Q-table, runs a greedy policy (no exploration), and
prints summary statistics including SLA-related info keys if present.
"""

import argparse

import numpy as np

from ran_slicing_qtable.action_mapper import ActionMapper
from ran_slicing_qtable.env_factory import make_env
from ran_slicing_qtable.q_table_agent import QTableAgent
from ran_slicing_qtable.reward_adapter import RewardAdapter
from ran_slicing_qtable.state_discretizer import StateDiscretizer
from ran_slicing_qtable.utils import gym_reset, gym_step, load_config, set_seed


def evaluate(config_path: str) -> None:
    cfg = load_config(config_path)

    seed: int = cfg.get("seed", 42)
    set_seed(seed)

    # --- Environment ---
    env_id: str = cfg["env_id"]
    print(f"[evaluate] Creating environment: {env_id}")
    env = make_env(env_id)

    # --- Observation dimension ---
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
    except Exception:
        obs0, _ = gym_reset(env, seed=seed)
        obs_dim = int(np.asarray(obs0).flatten().shape[0])
        print(f"[evaluate] Inferred obs_dim={obs_dim} from reset().")

    # --- Components ---
    n_bins: int = cfg.get("observation_bins", 5)
    discretizer = StateDiscretizer(obs_dim=obs_dim, n_bins=n_bins, mode="uniform")

    num_slices: int = cfg.get("num_slices", 3)
    max_rb: int = cfg.get("max_rb_per_slice", 10)
    action_mode: str = cfg.get("action_mode", "simple")
    mapper = ActionMapper(num_slices=num_slices, max_rb_per_slice=max_rb, mode=action_mode)

    adapter = RewardAdapter()

    agent = QTableAgent(n_actions=mapper.n_actions)
    q_table_path: str = cfg.get("q_table_path", "qtable_output.pkl")
    agent.load(q_table_path)

    # --- Greedy evaluation ---
    max_steps: int = cfg.get("max_steps_per_episode", 200)
    obs, _ = gym_reset(env, seed=seed)
    state = discretizer.discretize(obs)
    total_reward = 0.0
    step_count = 0
    sla_info_accumulated: dict = {}

    for step in range(max_steps):
        action_idx = agent.select_action(state, greedy=True)
        env_action = mapper.to_env_action(action_idx)
        obs_next, raw_reward, done, info = gym_step(env, env_action)

        reward = adapter.adapt(raw_reward, info=info, action_alloc=env_action)
        total_reward += reward
        step_count += 1

        # Accumulate any numeric info values for reporting.
        for key, val in info.items():
            if np.isscalar(val) or (isinstance(val, (list, np.ndarray))):
                prev = sla_info_accumulated.get(key, 0.0)
                try:
                    sla_info_accumulated[key] = float(prev) + float(np.sum(val))
                except (TypeError, ValueError):
                    pass  # skip non-numeric info fields

        state = discretizer.discretize(obs_next)
        if done:
            break

    env.close()

    # --- Report ---
    print("\n[evaluate] === Evaluation Results ===")
    print(f"  Total reward:  {total_reward:.4f}")
    print(f"  Steps taken:   {step_count}")
    if sla_info_accumulated:
        print("  Info totals:")
        for key, val in sla_info_accumulated.items():
            print(f"    {key}: {val:.4f}")
    print("======================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Q-table policy on the network-slicing env."
    )
    parser.add_argument(
        "--config",
        default="configs/qlearning.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
