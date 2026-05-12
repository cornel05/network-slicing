"""
train.py — Tabular Q-learning training loop for RAN network slicing.

Usage:
    python -m ran_slicing_qtable.train --config configs/qlearning.yaml

Prerequisites:
    1. This project is installed:  pip install -e .
    2. The original network-slicing Gym environment is installed separately.
       See: scripts/setup_network_slicing_env.sh
    3. configs/qlearning.yaml has the correct env_id set.

The script will exit with a clear error message if the environment
cannot be found.
"""

import argparse

import numpy as np
from tqdm import tqdm

from ran_slicing_qtable.action_mapper import ActionMapper
from ran_slicing_qtable.env_factory import make_env
from ran_slicing_qtable.q_table_agent import QTableAgent
from ran_slicing_qtable.reward_adapter import RewardAdapter
from ran_slicing_qtable.state_discretizer import StateDiscretizer
from ran_slicing_qtable.utils import ensure_dir, gym_reset, gym_step, load_config, set_seed


def train(config_path: str) -> None:
    cfg = load_config(config_path)

    # --- Reproducibility ---
    seed: int = cfg.get("seed", 42)
    set_seed(seed)

    # --- Environment ---
    env_id: str = cfg["env_id"]
    print(f"[train] Creating environment: {env_id}")
    env = make_env(env_id)

    # --- Determine observation dimensionality ---
    # TODO: After inspecting the original env, verify observation_space.shape.
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
    except Exception:
        # Fallback: attempt a reset and measure the observation.
        obs0, _ = gym_reset(env, seed=seed)
        obs_dim = int(np.asarray(obs0).flatten().shape[0])
        print(f"[train] Warning: could not read obs_dim from observation_space. "
              f"Inferred obs_dim={obs_dim} from reset().")

    # --- State discretizer ---
    n_bins: int = cfg.get("observation_bins", 5)
    # TODO: Replace None with env.observation_space.low / .high for tighter bins.
    discretizer = StateDiscretizer(obs_dim=obs_dim, n_bins=n_bins, mode="uniform")
    print(f"[train] {discretizer}")

    # --- Action mapper ---
    num_slices: int = cfg.get("num_slices", 3)
    max_rb: int = cfg.get("max_rb_per_slice", 10)
    action_mode: str = cfg.get("action_mode", "simple")
    mapper = ActionMapper(num_slices=num_slices, max_rb_per_slice=max_rb, mode=action_mode)
    mapper.describe()

    # --- Reward adapter ---
    adapter = RewardAdapter()

    # --- Q-table agent ---
    agent = QTableAgent(
        n_actions=mapper.n_actions,
        alpha=cfg.get("alpha", 0.1),
        gamma=cfg.get("gamma", 0.95),
        epsilon_start=cfg.get("epsilon_start", 1.0),
        epsilon_min=cfg.get("epsilon_min", 0.05),
        epsilon_decay=cfg.get("epsilon_decay", 0.995),
    )

    # --- Training loop ---
    n_episodes: int = cfg.get("episodes", 500)
    max_steps: int = cfg.get("max_steps_per_episode", 200)
    q_table_path: str = cfg.get("q_table_path", "qtable_output.pkl")
    ensure_dir(q_table_path)

    episode_rewards = []

    for episode in tqdm(range(n_episodes), desc="Training"):
        obs, _ = gym_reset(env, seed=None)  # only seed the first reset via set_seed
        state = discretizer.discretize(obs)
        total_reward = 0.0

        for _step in range(max_steps):
            # Select action index
            action_idx = agent.select_action(state)
            # Convert to environment-compatible allocation vector
            env_action = mapper.to_env_action(action_idx)

            # Step the environment
            # TODO: Verify that env_action format matches what the original env expects.
            obs_next, raw_reward, done, info = gym_step(env, env_action)

            # Adapt reward to scalar
            reward = adapter.adapt(raw_reward, info=info, action_alloc=env_action)
            total_reward += reward

            # Discretize next state
            next_state = discretizer.discretize(obs_next)

            # Q-table update
            agent.update(state, action_idx, reward, next_state, done)

            state = next_state
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        # Periodic progress log
        if (episode + 1) % 50 == 0:
            recent_mean = float(np.mean(episode_rewards[-50:]))
            print(
                f"  Episode {episode + 1}/{n_episodes} | "
                f"mean reward (last 50): {recent_mean:.3f} | "
                f"epsilon: {agent.epsilon:.4f} | "
                f"Q-table states: {len(agent._q)}"
            )

    # --- Save Q-table ---
    agent.save(q_table_path)
    env.close()
    print(f"\n[train] Training complete.  Q-table saved to {q_table_path}")
    print(f"[train] Final mean reward (last 50 eps): "
          f"{float(np.mean(episode_rewards[-50:])):.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-learning agent on the network-slicing env."
    )
    parser.add_argument(
        "--config",
        default="configs/qlearning.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
