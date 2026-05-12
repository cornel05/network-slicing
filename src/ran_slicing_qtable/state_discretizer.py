"""
state_discretizer.py — Convert raw observations into small discrete tuples.

The original network-slicing environment may return a continuous or
high-dimensional observation vector.  A raw tabular Q-table cannot handle
such a space directly, so we bin each dimension into a small number of
discrete levels.

TODO: Inspect the actual observation space returned by the original env
      (env.observation_space) to determine:
        - The number of dimensions (obs_dim).
        - The min/max range of each dimension.
        - Whether any dimensions are already integers.
      Then tune `n_bins` and the `low`/`high` bounds accordingly.
"""

import numpy as np


class StateDiscretizer:
    """
    Maps a raw observation vector to a small discrete tuple.

    Two modes are supported:
      - "uniform":  Divide each dimension into `n_bins` equal-width bins
                    using a user-supplied (low, high) range.
      - "passthrough": Treat the observation as already discrete; just
                    cast each element to int and wrap in a tuple.

    Keep `n_bins` small (e.g., 5) to limit the Q-table size.
    """

    def __init__(
        self,
        obs_dim: int,
        n_bins: int = 5,
        low: np.ndarray | None = None,
        high: np.ndarray | None = None,
        mode: str = "uniform",
    ):
        """
        Args:
            obs_dim:  Number of dimensions in the raw observation vector.
                      TODO: Set this to env.observation_space.shape[0].
            n_bins:   Number of discrete bins per dimension (uniform mode).
            low:      Lower bounds for each dimension.  If None, defaults to 0.
                      TODO: Set to env.observation_space.low.
            high:     Upper bounds for each dimension.  If None, defaults to 1.
                      TODO: Set to env.observation_space.high.
            mode:     "uniform" or "passthrough".
        """
        self.obs_dim = obs_dim
        self.n_bins = n_bins
        self.mode = mode

        if mode == "uniform":
            # Default to [0, 1] range if bounds are not provided.
            # Replace these with env.observation_space.low/high for accuracy.
            if low is None:
                low = np.zeros(obs_dim, dtype=np.float64)
            if high is None:
                high = np.ones(obs_dim, dtype=np.float64)

            self.low = np.asarray(low, dtype=np.float64)
            self.high = np.asarray(high, dtype=np.float64)

            # Pre-compute bin edges for each dimension.
            # np.digitize will assign each value to a bin in [1, n_bins].
            self._bin_edges = [
                np.linspace(self.low[i], self.high[i], n_bins + 1)
                for i in range(obs_dim)
            ]
        else:
            self.low = None
            self.high = None
            self._bin_edges = None

    def discretize(self, obs: np.ndarray) -> tuple:
        """
        Convert a raw observation array into a discrete state tuple.

        Args:
            obs:  1-D numpy array of shape (obs_dim,).

        Returns:
            A hashable tuple of integers, one per observation dimension.
        """
        obs = np.asarray(obs, dtype=np.float64).flatten()

        if self.mode == "passthrough":
            # Observation is already discrete (e.g., integer counts).
            return tuple(int(v) for v in obs)

        # Uniform binning: clip to [low, high] then digitize.
        obs_clipped = np.clip(obs, self.low, self.high)
        bins = []
        for i, edges in enumerate(self._bin_edges):
            # np.digitize returns values in [1, n_bins+1]; cap at n_bins.
            b = int(np.digitize(obs_clipped[i], edges[1:-1]))
            bins.append(min(b, self.n_bins - 1))
        return tuple(bins)

    def __repr__(self) -> str:
        return (
            f"StateDiscretizer(obs_dim={self.obs_dim}, "
            f"n_bins={self.n_bins}, mode={self.mode!r})"
        )
