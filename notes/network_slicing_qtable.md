# Design Notes: RAN Network Slicing — Tabular Q-Learning Baseline

## Problem Summary

The original [network-slicing environment](https://github.com/jjalcaraz-upct/network-slicing)
models RAN resource allocation as a sequential decision problem:

- **State**: A vector describing the current network load, queue occupancy,
  SLA status, and possibly channel quality for each slice.
- **Action**: The number of resource blocks (RBs) to allocate to each slice
  for the next observation period.
- **Reward**: An SLA fulfillment signal—positive when SLAs are met,
  negative or zero when violated.
- **Objective**: Minimise RB usage while satisfying all SLA constraints.

---

## State / Action / Reward Mapping

### State
- The raw observation vector may contain continuous values (load levels,
  queue sizes) and integer values (SLA violation flags).
- We flatten and discretize this vector into a small tuple of integers using
  `StateDiscretizer`.
- Number of bins per dimension is kept at 5 (configurable) to limit Q-table
  growth.

### Action
- The full action space is all non-negative integer allocations summing to
  at most `total_RB_budget` across N slices.  For 3 slices and 10 RBs each,
  this is already in the thousands.
- We replace it with ~6–8 hand-crafted strategies via `ActionMapper`:
  balanced, per-slice priority, conservative, and aggressive.
- The Q-table learns which strategy to apply in each discretized state.

### Reward
- The adapter prefers to use the raw scalar reward directly.
- Optional penalties for SLA violations and total RB usage can be enabled
  by setting the corresponding weights in `RewardAdapter`.

---

## Why This Is Harder Than CliffWalking

| Property              | CliffWalking     | RAN Slicing               |
|-----------------------|------------------|---------------------------|
| State space           | Small (4×12 grid) | Large continuous vector  |
| Action space          | 4 directions     | Combinatorial RB splits   |
| Reward signal         | Dense (−1/step)  | Sparse SLA signals        |
| Stationarity          | Deterministic    | Stochastic traffic load   |
| Episode length        | Short (~13 steps)| Many steps per episode    |

Tabular Q-learning converges in CliffWalking because the state×action table
is small enough to visit every cell many times.  In RAN slicing, without
discretization and action reduction, the table would never converge.

---

## Why Discretization Is Required

A Q-table stores one value per `(state, action)` pair.  If states are
continuous or high-dimensional:

- Identical physical situations map to different float vectors → never revisit.
- Table grows without bound, memory explodes, estimates stay near zero.

Binning each dimension into 5 levels with 6 observation dimensions gives
5⁶ = 15,625 states.  With ~8 actions, that is ~125,000 Q-values—still
manageable for a lightweight baseline.

---

## Why Action-Space Reduction Is Required

For 3 slices with up to 10 RBs each, the number of distinct allocations is:
C(10+3, 3) ≈ 286 if budgeted, more if unconstrained.

Larger configurations grow much faster.  Exploring 286+ actions per state
requires enormous sample complexity for tabular methods.  The hand-crafted
action set reduces this to ~8 meaningful strategies, losing some optimality
but making learning tractable.

---

## What Can Break in Practice

1. **Wrong `env_id`**: The Gym registration string must be verified from the
   original repo.  Training will fail immediately with a clear error.

2. **Wrong action format**: The original env may expect a flat integer, a
   dict, or a differently shaped numpy array.  Check `env.action_space` and
   update `ActionMapper.to_env_action()`.

3. **Observation bounds mismatch**: If the real obs range differs from `[0, 1]`,
   all observations fall in the same bin and the agent cannot learn.  Update
   `StateDiscretizer` with `env.observation_space.low` / `.high`.

4. **Sparse reward**: If the env only rewards at episode end, many episodes
   will have zero gradient.  Consider reward shaping.

5. **Legacy dependency conflicts**: gym==0.15.3 conflicts with modern numpy/pandas.
   Use the dedicated legacy venv from `setup_network_slicing_env.sh`.

---

## Next Steps After the Baseline Works

1. Tune `observation_bins`, `alpha`, `gamma`, and the action set based on
   learning curves.
2. Replace hand-crafted actions with a finer enumerated set once the env's
   RB budget is known.
3. Add reward shaping based on per-slice SLA violation signals from `info`.
4. Compare against a random policy to confirm the agent is learning.
5. Consider adding a simple neural network (DQN) once the tabular baseline
   is validated—only if deeper exploration is needed.
