"""
action_mapper.py — Map a small discrete action index to an RB allocation vector.

The original network-slicing environment expects an action that specifies the
number of resource blocks (RBs) allocated to each slice.  The full action
space is combinatorial (all ways to distribute RBs across slices), which is
far too large for tabular Q-learning.

Instead, we define a small, hand-crafted set of allocation strategies.
The Q-table learns which strategy to apply in each discretized state.

TODO: After inspecting the original environment, verify:
  - The exact format of the action (numpy array, dict, or flat int?).
  - The total RB budget constraint, if any.
  - Whether actions are per-slice allocations or ratios.
  Then update `_build_actions()` accordingly.
"""

import numpy as np


class ActionMapper:
    """
    Maps a small integer index to an RB allocation vector.

    Supported mode: "simple"
      A fixed set of hand-crafted strategies covering:
        - Balanced allocation across all slices.
        - Prioritise each slice in turn.
        - Conservative (low RB) allocation.
        - Aggressive (high RB) allocation.

    Attributes:
        n_actions (int):  Total number of available strategies.
    """

    def __init__(
        self,
        num_slices: int = 3,
        max_rb_per_slice: int = 10,
        mode: str = "simple",
    ):
        """
        Args:
            num_slices:       Number of network slices.
            max_rb_per_slice: Maximum RBs that can be assigned to a single slice.
            mode:             Strategy set to use. Currently only "simple".
        """
        self.num_slices = num_slices
        self.max_rb_per_slice = max_rb_per_slice
        self.mode = mode

        self._actions = self._build_actions()
        self.n_actions = len(self._actions)

    def _build_actions(self) -> list[np.ndarray]:
        """
        Construct the list of allocation vectors.

        Each vector has shape (num_slices,) with integer RB counts.
        The strategies are deliberately simple and human-interpretable.
        """
        mid = self.max_rb_per_slice // 2
        low = max(1, self.max_rb_per_slice // 4)
        high = self.max_rb_per_slice

        actions = []

        # 1. Balanced: give every slice the same (mid) allocation.
        actions.append(np.full(self.num_slices, mid, dtype=np.int32))

        # 2. Prioritise each slice in turn (one slice gets max, others get low).
        for i in range(self.num_slices):
            alloc = np.full(self.num_slices, low, dtype=np.int32)
            alloc[i] = high
            actions.append(alloc)

        # 3. Conservative: all slices get the low allocation.
        actions.append(np.full(self.num_slices, low, dtype=np.int32))

        # 4. Aggressive: all slices get the high allocation.
        actions.append(np.full(self.num_slices, high, dtype=np.int32))

        # 5. Half-and-half for 2-slice subsets (only when num_slices >= 2).
        if self.num_slices >= 2:
            alloc = np.full(self.num_slices, low, dtype=np.int32)
            alloc[0] = high
            alloc[1] = high
            actions.append(alloc.copy())

        return actions

    def to_env_action(self, action_index: int) -> np.ndarray:
        """
        Convert a Q-table action index to an environment-compatible allocation.

        Args:
            action_index:  Integer in [0, n_actions).

        Returns:
            Numpy integer array of shape (num_slices,) with RB counts.

        TODO: The original environment may require a different action format
              (e.g., a flat integer or a dict).  Adapt the return value here
              once the action space has been inspected.
        """
        if not (0 <= action_index < self.n_actions):
            raise ValueError(
                f"action_index {action_index} out of range [0, {self.n_actions})"
            )
        return self._actions[action_index].copy()

    def describe(self) -> None:
        """Print all available actions for debugging."""
        print(f"ActionMapper: {self.n_actions} actions, "
              f"num_slices={self.num_slices}, max_rb={self.max_rb_per_slice}")
        for i, a in enumerate(self._actions):
            print(f"  [{i}] {a}")

    def __repr__(self) -> str:
        return (
            f"ActionMapper(n_actions={self.n_actions}, "
            f"num_slices={self.num_slices}, max_rb_per_slice={self.max_rb_per_slice})"
        )
