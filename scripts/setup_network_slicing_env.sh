#!/usr/bin/env bash
# =============================================================================
# setup_network_slicing_env.sh
#
# Documents the setup flow for the ORIGINAL network-slicing Gym environment:
#   https://github.com/jjalcaraz-upct/network-slicing
#
# This script sets up the original environment in a SEPARATE virtual
# environment to avoid dependency conflicts with this Q-table project.
#
# WARNING: The original repo uses legacy dependencies:
#   gym==0.15.3, numpy==1.19.1, pandas==0.25.2
#
# These are old and may conflict with modern Python packages. A separate
# environment is strongly recommended.
#
# NOTE: TensorFlow, Stable-Baselines v2, Keras, and Keras-RL are NOT
# installed by this script. Those are only needed to reproduce the original
# paper agents (PPO, DQN). This Q-table baseline does NOT require them.
# =============================================================================

set -euo pipefail

INSTALL_DIR="${1:-./network_slicing_env}"
VENV_DIR="${INSTALL_DIR}/.venv_legacy"

echo "============================================================"
echo "  Setting up the original network-slicing Gym environment"
echo "  Install directory: ${INSTALL_DIR}"
echo "============================================================"
echo ""

# --- Step 1: Clone the original repository ---
echo "[1/4] Cloning https://github.com/jjalcaraz-upct/network-slicing ..."
if [ -d "${INSTALL_DIR}/network-slicing" ]; then
    echo "      Directory already exists, skipping clone."
else
    git clone https://github.com/jjalcaraz-upct/network-slicing \
        "${INSTALL_DIR}/network-slicing"
fi
echo ""

# --- Step 2: Create a legacy virtual environment ---
echo "[2/4] Creating a legacy Python virtual environment at ${VENV_DIR} ..."
echo "      (Keeps old dependencies isolated from this Q-table project)"
python3 -m venv "${VENV_DIR}"
echo ""

# --- Step 3: Install the Gym environment package ---
echo "[3/4] Installing the network-slicing Gym package (gym-ran_slice) ..."
echo ""
echo "      WARNING: The original repo requires:"
echo "        gym==0.15.3"
echo "        numpy==1.19.1"
echo "        pandas==0.25.2"
echo "      These are legacy versions. Installing into the isolated venv."
echo ""

# Install the Gym env package from the cloned repo
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install \
    "gym==0.15.3" \
    "numpy==1.19.1" \
    "pandas==0.25.2"

# The Gym environment package lives inside gym-ran_slice/
"${VENV_DIR}/bin/pip" install -e "${INSTALL_DIR}/network-slicing/gym-ran_slice"

echo ""
echo "[4/4] Verifying installation ..."
"${VENV_DIR}/bin/python" -c "
import gym_ran_slice
import gym
envs = [spec.id for spec in gym.envs.registry.all() if 'ran' in spec.id.lower() or 'slice' in spec.id.lower()]
print('Registered RAN/slice environments:', envs)
print('gym_ran_slice import OK')
" || echo "WARNING: Verification failed. Check the error above."

echo ""
echo "============================================================"
echo "  Setup complete!"
echo ""
echo "  To use the original env from this Q-table baseline,"
echo "  activate the legacy venv before running training:"
echo ""
echo "    source ${VENV_DIR}/bin/activate"
echo "    python -m ran_slicing_qtable.train --config configs/qlearning.yaml"
echo ""
echo "  Then update configs/qlearning.yaml with the correct env_id."
echo ""
echo "  NOTE: TensorFlow, Stable-Baselines, Keras, and Keras-RL are NOT"
echo "  installed. They are only needed for the original paper agents."
echo "============================================================"
