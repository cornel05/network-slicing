"""
reward_adapter.py — Convert raw environment reward/info into a scalar.

The original network-slicing environment may return:
  - A scalar reward (ideal case — used directly).
  - A reward vector or dict (needs reduction).
  - Additional info dict with SLA violation flags per slice.

This adapter provides optional penalties on top of the base reward to
encourage SLA compliance and efficient RB usage.

TODO: After inspecting the original environment's step() output, check
  - The type and shape of the raw reward.
  - Which keys in `info` carry SLA violation signals.
  Then tune the penalty weights below.
"""

import numpy as np


class RewardAdapter:
    """
    Converts raw (reward, info, action) from env.step() into a scalar reward.

    Optional penalties:
      - sla_penalty_weight:  Multiplied by the number of violated SLAs.
      - rb_penalty_weight:   Multiplied by total RBs used (encourages efficiency).
    """

    def __init__(
        self,
        sla_penalty_weight: float = 0.0,
        rb_penalty_weight: float = 0.0,
        # TODO: Set the correct info key for SLA violations after env inspection.
        sla_violation_key: str = "sla_violation",
    ):
        """
        Args:
            sla_penalty_weight:  Penalty per violated SLA (set > 0 to enable).
            rb_penalty_weight:   Penalty per RB used (set > 0 to enable).
            sla_violation_key:   Key in `info` dict that holds SLA violation data.
                                 Could be a list of booleans or a scalar count.
        """
        self.sla_penalty_weight = sla_penalty_weight
        self.rb_penalty_weight = rb_penalty_weight
        self.sla_violation_key = sla_violation_key

    def adapt(
        self,
        reward,
        info: dict | None = None,
        action_alloc: np.ndarray | None = None,
    ) -> float:
        """
        Compute the final scalar reward for Q-learning.

        Args:
            reward:        Raw reward from env.step().  Scalar, array, or list.
            info:          Info dict returned alongside the reward.
            action_alloc:  RB allocation vector (from ActionMapper.to_env_action),
                           used for the optional RB-usage penalty.

        Returns:
            A single float suitable for a Q-table update.
        """
        # --- Base reward ---
        if np.isscalar(reward):
            scalar_reward = float(reward)
        else:
            # If reward is a vector, sum it as a simple fallback.
            # TODO: Verify the original reward structure and update this logic.
            scalar_reward = float(np.sum(reward))

        # --- Optional SLA violation penalty ---
        if self.sla_penalty_weight > 0.0 and info is not None:
            violations = info.get(self.sla_violation_key, None)
            if violations is not None:
                if np.isscalar(violations):
                    n_violations = float(violations)
                else:
                    # Assume a boolean/int array: count the True values.
                    n_violations = float(np.sum(violations))
                scalar_reward -= self.sla_penalty_weight * n_violations

        # --- Optional RB usage penalty ---
        if self.rb_penalty_weight > 0.0 and action_alloc is not None:
            total_rb = float(np.sum(action_alloc))
            scalar_reward -= self.rb_penalty_weight * total_rb

        return scalar_reward

    def __repr__(self) -> str:
        return (
            f"RewardAdapter(sla_penalty={self.sla_penalty_weight}, "
            f"rb_penalty={self.rb_penalty_weight})"
        )
