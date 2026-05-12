"""
q_table_agent.py — Tabular Q-learning agent.

Stores Q-values in a Python dict keyed by (discrete_state_tuple, action_index).
Works well when the state space is small (after discretization).
"""

import pickle
from pathlib import Path

import numpy as np


class QTableAgent:
    """
    A simple tabular Q-learning agent.

    States are expected to be hashable tuples (produced by StateDiscretizer).
    Actions are small integers in [0, n_actions).
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        """
        Args:
            n_actions:      Number of discrete actions (from ActionMapper).
            alpha:          Learning rate.
            gamma:          Discount factor.
            epsilon_start:  Initial exploration probability.
            epsilon_min:    Minimum exploration probability.
            epsilon_decay:  Multiplicative decay applied each episode.
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: maps discrete state tuple -> numpy array of shape (n_actions,)
        # Entries are created lazily (defaulting to zeros) the first time a state
        # is encountered.
        self._q: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_q(self, state: tuple) -> np.ndarray:
        """Return the Q-value array for a state, initializing to zeros if new."""
        if state not in self._q:
            self._q[state] = np.zeros(self.n_actions, dtype=np.float64)
        return self._q[state]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state: tuple, greedy: bool = False) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state:   Discrete state tuple from StateDiscretizer.
            greedy:  If True, always pick the best known action (no exploration).

        Returns:
            Action index in [0, n_actions).
        """
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        q_values = self._get_q(state)
        return int(np.argmax(q_values))

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        """
        Apply one Q-learning update:
            Q(s, a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

        If done is True, the future value term is zeroed out.
        """
        q_current = self._get_q(state)[action]
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self._get_q(next_state))
        td_error = td_target - q_current
        self._get_q(state)[action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Reduce epsilon by the decay factor, respecting the minimum."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Persist the Q-table and agent parameters to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_table": self._q,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"[QTableAgent] Q-table saved to {path} ({len(self._q)} states)")

    def load(self, path: str) -> None:
        """Load a previously saved Q-table and parameters from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._q = data["q_table"]
        self.n_actions = data["n_actions"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
        self.epsilon = data["epsilon"]
        self.epsilon_min = data["epsilon_min"]
        self.epsilon_decay = data["epsilon_decay"]
        print(f"[QTableAgent] Q-table loaded from {path} ({len(self._q)} states)")
