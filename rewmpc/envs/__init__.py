"""
Environment factory for ReW-MPC.

Adapted from TD-MPC2 (Hansen et al., 2024) — https://github.com/nicklashansen/tdmpc2
Original licence: MIT. Changes: relative imports, MetaWorld-only, gymnasium compat.
"""
from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper
from envs.metaworld import make_env as make_metaworld_env

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_multitask_env(cfg):
	"""Make a multi-task environment."""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_env(_cfg)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env


def make_env(cfg):
	"""Make an environment. Supports MetaWorld MT10 (state-based)."""
	if getattr(cfg, 'multitask', False):
		env = make_multitask_env(cfg)
	else:
		try:
			env = make_metaworld_env(cfg)
		except ValueError:
			raise ValueError(
				f'Failed to make environment "{cfg.task}": check that metaworld is installed '
				f'and the task name is valid (expected mw-<name>).'
			)
		env = TensorWrapper(env)

	try:
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except Exception:
		obs_key = cfg.obs if hasattr(cfg, 'obs') else 'state'
		cfg.obs_shape = {obs_key: env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = getattr(env, 'max_episode_steps', 100)
	cfg.seed_steps = max(1000, 5 * cfg.episode_length)
	return env
