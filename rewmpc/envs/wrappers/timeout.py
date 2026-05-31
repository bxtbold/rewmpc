"""
Episode timeout wrapper for ReW-MPC.

Adapted from TD-MPC2 (Hansen et al., 2024) — https://github.com/nicklashansen/tdmpc2
Original licence: MIT. No functional changes.
"""
import gymnasium as gym


class Timeout(gym.Wrapper):
	"""Enforces a hard step limit per episode."""

	def __init__(self, env, max_episode_steps):
		super().__init__(env)
		self._max_episode_steps = max_episode_steps

	@property
	def max_episode_steps(self):
		return self._max_episode_steps

	def reset(self, **kwargs):
		self._t = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._t += 1
		done = done or self._t >= self.max_episode_steps
		return obs, reward, done, info
