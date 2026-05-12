#!/usr/bin/env bash
# =============================================================================
# run_train.sh
#
# Launch the tabular Q-learning training loop.
#
# Prerequisites:
#   1. This project is installed:  pip install -e .
#   2. The original network-slicing Gym environment is installed separately.
#      See: scripts/setup_network_slicing_env.sh
#   3. configs/qlearning.yaml has the correct env_id set.
# =============================================================================

set -euo pipefail

CONFIG="${1:-configs/qlearning.yaml}"

echo "Starting Q-table training with config: ${CONFIG}"
python -m ran_slicing_qtable.train --config "${CONFIG}"
