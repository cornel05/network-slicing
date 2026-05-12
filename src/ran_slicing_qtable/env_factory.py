"""
env_factory.py — Create the network-slicing Gym environment.

The original environment lives in a separate repository and must be
installed before this factory can work:
  https://github.com/jjalcaraz-upct/network-slicing

This module tries to import gym and create the environment, printing
clear error messages if something is missing.

It does NOT fake an environment.  If the env cannot be created, the
process exits with an informative message.
"""


def make_env(env_id: str):
    """
    Create and return the network-slicing Gym environment.

    Args:
        env_id:  The Gym registration string, e.g. "ran-slicing-v0".
                 TODO: Verify the correct ID by running:
                   python -c "import gym_ran_slice; import gym; \\
                              [print(s.id) for s in gym.envs.registry.all()]"
                 after installing the original environment.

    Returns:
        An OpenAI Gym (or Gymnasium) environment instance.

    Raises:
        SystemExit if gym or the environment cannot be imported/created.
    """
    # --- Try importing gym ---
    try:
        import gym  # type: ignore[import]
    except ImportError:
        print(
            "\n[env_factory] ERROR: 'gym' is not installed.\n"
            "  The original network-slicing environment requires gym==0.15.3.\n"
            "  Install it inside the legacy virtual environment:\n"
            "    pip install gym==0.15.3\n"
            "  See scripts/setup_network_slicing_env.sh for the full setup.\n"
        )
        raise SystemExit(1)

    # --- Try importing the env registration module ---
    try:
        import gym_ran_slice  # type: ignore[import]  # noqa: F401
    except ImportError:
        print(
            "\n[env_factory] ERROR: 'gym_ran_slice' is not installed.\n"
            "  The network-slicing Gym environment package must be installed "
            "separately from:\n"
            "    https://github.com/jjalcaraz-upct/network-slicing\n"
            "  Run: bash scripts/setup_network_slicing_env.sh\n"
            "  Then activate the legacy virtual environment before training.\n"
        )
        raise SystemExit(1)

    # --- Try creating the environment ---
    try:
        env = gym.make(env_id)
    except Exception as exc:
        print(
            f"\n[env_factory] ERROR: Could not create environment '{env_id}'.\n"
            f"  Gym error: {exc}\n"
            f"\n"
            f"  Possible causes:\n"
            f"    1. The env_id is wrong.  Check configs/qlearning.yaml.\n"
            f"       Run the following to list registered envs:\n"
            f"         python -c \"import gym_ran_slice, gym; "
            f"[print(s.id) for s in gym.envs.registry.all()]\"\n"
            f"    2. The gym_ran_slice package was not installed correctly.\n"
            f"       Re-run: bash scripts/setup_network_slicing_env.sh\n"
        )
        raise SystemExit(1)

    return env
