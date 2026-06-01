"""
MetaWorld environment wrapper for ReW-MPC.

Adapted from TD-MPC2 (Hansen et al., 2024) — https://github.com/nicklashansen/tdmpc2
Original licence: MIT.
Changes: updated for MetaWorld 3.0 / gymnasium API (v3 envs, 5-tuple step,
2-tuple reset, MT10 task-class lookup).
"""
import numpy as np
import gymnasium as gym

from envs.wrappers.timeout import Timeout

import metaworld


# Map mw-<name> → <name>-v3 (MetaWorld 3.0 task names)
def _task_id(cfg_task):
	return cfg_task.split('-', 1)[-1] + '-v3'


# Build a shared MT10 benchmark instance once per process.
_MT10_CACHE = {}


def _get_mt10(seed):
	if seed not in _MT10_CACHE:
		_MT10_CACHE[seed] = metaworld.MT10(seed=seed)
	return _MT10_CACHE[seed]


class MetaWorldWrapper(gym.Wrapper):
	def __init__(self, env, tasks, cfg):
		super().__init__(env)
		self._tasks = tasks
		self._rng = np.random.default_rng(cfg.seed)
		self._cfg = cfg

	def _sample_task(self):
		idx = self._rng.integers(len(self._tasks))
		return self._tasks[idx]

	def reset(self, **kwargs):
		self.env.set_task(self._sample_task())
		obs, _ = self.env.reset(**kwargs)
		return obs.astype(np.float32)

	def step(self, action):
		reward = 0.0
		for _ in range(2):
			obs, r, terminated, truncated, info = self.env.step(action.copy())
			reward += r
		obs = obs.astype(np.float32)
		done = terminated or truncated
		return obs, reward, done, info

	@property
	def unwrapped(self):
		return self.env.unwrapped


def make_env(cfg):
	"""Make a MetaWorld v3 environment."""
	task_id = _task_id(cfg.task)
	if not cfg.task.startswith('mw-'):
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'

	mt10 = _get_mt10(cfg.seed)
	if task_id not in mt10.train_classes:
		raise ValueError(f'Task {task_id} not found in MT10. Available: {list(mt10.train_classes)}')

	EnvCls = mt10.train_classes[task_id]
	render_mode = 'rgb_array' if getattr(cfg, 'save_video', False) else None
	raw_env = EnvCls() if render_mode is None else EnvCls(render_mode=render_mode)
	tasks = [t for t in mt10.train_tasks if t.env_name == task_id]

	env = MetaWorldWrapper(raw_env, tasks, cfg)
	env = Timeout(env, max_episode_steps=100)
	return env
