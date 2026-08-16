"""
Point mass environment wrapper for the ReW-MPC pipeline.

Adapts toys/point_mass_env.py to the same interface as MetaWorldWrapper so
train.py / evaluate.py work unchanged with task=pm-2d.
"""
import sys
import os
from pathlib import Path

import numpy as np
import gymnasium as gym

from envs.wrappers.timeout import Timeout

# Resolve toys/ relative to this file's location
_TOYS_DIR = str(Path(__file__).parent.parent.parent / 'toys')


def _import_pm_env():
    if _TOYS_DIR not in sys.path:
        sys.path.insert(0, _TOYS_DIR)
    from point_mass_env import PointMassEnv
    return PointMassEnv


class PointMassWrapper(gym.Wrapper):
    """
    Thin shim around PointMassEnv that matches the MetaWorldWrapper interface:
      - reset()  → obs (numpy float32)
      - step(a)  → (obs, reward, done, info)  [4-tuple, not gymnasium 5-tuple]
      - info must contain 'success' (float) and 'terminated' (float)
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs.astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info['terminated'] = float(terminated)
        if 'success' not in info:
            info['success'] = 0.0
        return obs.astype(np.float32), float(reward), done, info


def make_env(cfg):
    """Make a PointMass environment and populate cfg fields."""
    PointMassEnv = _import_pm_env()

    N           = getattr(cfg, 'pm_N', 2)
    max_steps   = getattr(cfg, 'pm_max_steps', 50)
    goal        = list(getattr(cfg, 'pm_goal', [1.0] + [0.0] * (N - 1)))
    sigma       = getattr(cfg, 'pm_reward_sigma', 0.3)
    success_dist = getattr(cfg, 'pm_success_dist', 0.15)

    raw_env = PointMassEnv(
        N=N, dt=0.05, max_steps=max_steps,
        goal=goal, reward_sigma=sigma, success_dist=success_dist,
    )
    env = PointMassWrapper(raw_env)
    env = Timeout(env, max_episode_steps=max_steps)
    return env
