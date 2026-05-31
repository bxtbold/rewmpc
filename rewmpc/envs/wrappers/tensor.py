"""
Numpy → torch tensor wrapper for ReW-MPC.

Adapted from TD-MPC2 (Hansen et al., 2024) — https://github.com/nicklashansen/tdmpc2
Original licence: MIT. No functional changes.
"""
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch


class TensorWrapper(gym.Wrapper):
	"""Converts numpy arrays to torch tensors at the env boundary."""

	def __init__(self, env):
		super().__init__(env)

	@property
	def max_episode_steps(self):
		return getattr(self.env, 'max_episode_steps', None)

	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _try_f32_tensor(self, x):
		if isinstance(x, np.ndarray):
			x = torch.from_numpy(x)
			if x.dtype == torch.float64:
				x = x.float()
		return x

	def _obs_to_tensor(self, obs):
		if isinstance(obs, dict):
			for k in obs.keys():
				obs[k] = self._try_f32_tensor(obs[k])
		else:
			obs = self._try_f32_tensor(obs)
		return obs

	def reset(self, task_idx=None):
		return self._obs_to_tensor(self.env.reset())

	def step(self, action):
		obs, reward, done, info = self.env.step(action.numpy())
		info = defaultdict(float, info)
		info['success'] = float(info['success'])
		info['terminated'] = torch.tensor(float(info['terminated']))
		return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), done, info
