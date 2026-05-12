"""
utils.py — Shared utilities for the ran_slicing_qtable package.

Covers:
  - YAML config loading
  - Random seed initialization
  - Directory creation
  - Compatibility shims for old (gym 0.15.x) and new (gym 0.26+ / Gymnasium)
    reset() and step() return signatures.
"""

import os
import random
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file and return it as a dict.

    Args:
        path:  Path to the YAML file (e.g., "configs/qlearning.yaml").

    Returns:
        Dictionary of configuration values.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy to encourage reproducibility.

    Note: The Gym environment itself may have its own seed mechanism.
    Call env.seed(seed) or env.reset(seed=seed) separately if needed.

    Args:
        seed:  Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    """
    Create the parent directory of `path` if it does not already exist.

    Useful before saving files (Q-tables, logs, etc.).

    Args:
        path:  File path whose parent directory should be created.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Gym API compatibility shims
# ---------------------------------------------------------------------------

def gym_reset(env, seed: int | None = None):
    """
    Call env.reset() in a way that works with both old and new Gym APIs.

    Old Gym (<=0.25):  reset() returns obs (numpy array).
    New Gym (>=0.26) / Gymnasium:  reset() returns (obs, info).

    Args:
        env:   A Gym environment instance.
        seed:  Optional seed to pass to new-style reset().

    Returns:
        obs:   The initial observation (numpy array).
        info:  Info dict (empty dict for old-style envs).
    """
    try:
        # New-style: reset accepts keyword arguments
        result = env.reset(seed=seed)
    except TypeError:
        # Old-style: reset() takes no arguments
        result = env.reset()

    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        obs = result
        info = {}

    return obs, info


def gym_step(env, action):
    """
    Call env.step(action) in a way that works with both old and new Gym APIs.

    Old Gym (<=0.25):  step() returns (obs, reward, done, info).
    New Gym (>=0.26) / Gymnasium:  step() returns (obs, reward, terminated, truncated, info).

    Args:
        env:     A Gym environment instance.
        action:  The action to take.

    Returns:
        obs:       Next observation.
        reward:    Scalar (or array) reward.
        done:      True if the episode has ended for any reason.
        info:      Info dict from the environment.
    """
    result = env.step(action)

    if len(result) == 5:
        # New-style: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        # Old-style: (obs, reward, done, info)
        obs, reward, done, info = result

    return obs, reward, done, info
