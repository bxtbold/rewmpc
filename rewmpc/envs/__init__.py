"""
Environment factory for ReW-MPC. Delegates to TD-MPC2's environment wrappers,
which support DMControl, Meta-World, ManiSkill, and MyoSuite.
"""
import sys
import os

# Expose TD-MPC2's env wrappers without copying them
_tdmpc2_path = os.path.join(os.path.dirname(__file__), '../../../tdmpc2/tdmpc2')
if _tdmpc2_path not in sys.path:
	sys.path.insert(0, _tdmpc2_path)

from envs import make_env  # noqa: F401 — re-export TD-MPC2's make_env
