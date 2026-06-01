import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from trainer.base import Trainer


class OfflineTrainer(Trainer):
	"""
	3-phase offline trainer for ReW-MPC.

	Phase 1:  World model (JEPA + SigReg): trains encoder, GRU, dynamics.
	Phase 2:  Surrogates: trains reward and value heads (encoder frozen via stop-grad).
	Phase 3:  Flow prior: trains return-weighted flow (everything else frozen).
	          Epochs 0..bootstrap_epochs: uniform weights (pure BC warm-up).
	          Epochs bootstrap_epochs+: return-weighted CFM.

	Gradient-based planning evaluation runs during and after Phase 3.
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()

	def eval(self):
		"""Evaluate the agent on all tasks using gradient-based planning."""
		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				while not done:
					action = self.agent.act(obs, t0=(t == 0), eval_mode=True, task=task_idx)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info.get('success', 0.))
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),
			})
		return results

	def _load_dataset(self):
		"""Load TD-MPC2-compatible offline dataset."""
		if self.cfg.task.startswith('mw-'):
			# Single-task: look for <task>.pt in data_dir
			fps = [str(Path(self.cfg.data_dir) / f'{self.cfg.task}.pt')]
			assert Path(fps[0]).exists(), f'No data found at {fps[0]}'
		else:
			fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
			fps = sorted(glob(str(fp)))
			assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {self.cfg.data_dir}')
		n_expected = 20 if self.cfg.task == 'mt80' else 4
		if not self.cfg.task.startswith('mw-') and len(fps) < n_expected:
			print(f'WARNING: expected {n_expected} files, found {len(fps)}.')

		_cfg = deepcopy(self.cfg)
		# MetaWorld tasks (mt10, mt80, and single mw-*) use 101-step episodes
		_is_mw = self.cfg.task in {'mt10', 'mt80'} or self.cfg.task.startswith('mw-')
		_cfg.episode_length = 101 if _is_mw else 501
		_cfg.buffer_size = (
			550_450_000 if self.cfg.task == 'mt80' else
			50_000_000  if self.cfg.task == 'mt10' else
			5_000_000   if self.cfg.task.startswith('mw-') else
			345_690_000
		)
		_cfg.steps = _cfg.buffer_size
		self.buffer = Buffer(_cfg)
		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp, weights_only=False)
			assert td.shape[1] == _cfg.episode_length, \
				f'Episode length mismatch: got {td.shape[1]}, expected {_cfg.episode_length}'
			self.buffer.load(td)
		print(f'Dataset loaded: {self.buffer.num_eps} episodes.')

	def _sample_batch(self):
		"""Sample a training batch and return (obs, action, reward, task)."""
		obs, action, reward, _terminated, task = self.buffer.sample()
		return obs, action, reward, task

	def _load_phase(self, identifier):
		"""Load a saved phase checkpoint from the run's model directory."""
		fp = self.logger.model_dir / f'{identifier}.pt'
		assert fp.exists(), f'Checkpoint not found: {fp}'
		self.agent.load(str(fp))
		print(f'Loaded checkpoint: {fp}')

	def train(self):
		"""Run the 3-phase training curriculum."""
		assert self.cfg.task in {'mt10', 'mt30', 'mt80'} or self.cfg.task.startswith('mw-'), \
			'Offline training requires task=mt10/mt30/mt80 or a single mw-<task>.'
		self._load_dataset()
		metrics = {}
		resume_phase = getattr(self.cfg, 'resume_phase', 0)

		# ---- Phase 1: World model ----
		if resume_phase >= 2:
			print('[Phase 1] Skipped — loading saved checkpoint.')
			self._load_phase('phase1')
		else:
			print(f'\n[Phase 1] Training world model for {self.cfg.phase1_steps} steps...')
			for i in tqdm(range(self.cfg.phase1_steps), desc='Phase 1'):
				obs, action, _reward, task = self._sample_batch()
				train_metrics = self.agent.update_world(obs, action, task)

				if i % 5_000 == 0:
					metrics = {
						'iteration': i,
						'elapsed_time': time() - self._start_time,
					}
					metrics.update({k: v.item() for k, v in train_metrics.items()})
					self.logger.log(metrics, 'pretrain')

			self.logger.save_agent(self.agent, identifier='phase1')
			print('[Phase 1] Complete.')

		# ---- Phase 2: Surrogates ----
		if resume_phase >= 3:
			print('[Phase 2] Skipped — loading saved checkpoint.')
			self._load_phase('phase2')
		else:
			print(f'\n[Phase 2] Training reward + value surrogates for {self.cfg.phase2_steps} steps...')
			for i in tqdm(range(self.cfg.phase2_steps), desc='Phase 2'):
				obs, action, reward, task = self._sample_batch()
				train_metrics = self.agent.update_surrogates(obs, action, reward, task)

				if i % 5_000 == 0:
					metrics = {
						'iteration': self.cfg.phase1_steps + i,
						'elapsed_time': time() - self._start_time,
					}
					metrics.update({k: v.item() for k, v in train_metrics.items()})
					self.logger.log(metrics, 'pretrain')

			self.logger.save_agent(self.agent, identifier='phase2')
			print('[Phase 2] Complete.')

		# ---- Phase 3: Flow prior ----
		print(f'\n[Phase 3] Training flow prior for {self.cfg.phase3_steps} steps...')
		bootstrap_steps = int(self.cfg.flow_bootstrap_frac * self.cfg.phase3_steps)
		for i in tqdm(range(self.cfg.phase3_steps), desc='Phase 3'):
			obs, action, reward, task = self._sample_batch()
			use_weights = (i >= bootstrap_steps)
			train_metrics = self.agent.update_flow(obs, action, reward, task, use_weights=use_weights)

			step_global = self.cfg.phase1_steps + self.cfg.phase2_steps + i
			if i > 0 and i % self.cfg.eval_freq == 0:
				eval_metrics = self.eval()
				metrics = {
					'iteration': step_global,
					'elapsed_time': time() - self._start_time,
				}
				metrics.update({k: v.item() for k, v in train_metrics.items()})
				metrics.update(eval_metrics)
				self.logger.pprint_multitask(metrics, self.cfg)
				self.logger.log(metrics, 'pretrain')
				if i > 0:
					self.logger.save_agent(self.agent, identifier=f'phase3_{i}')
			elif i % 5_000 == 0:
				metrics = {
					'iteration': step_global,
					'elapsed_time': time() - self._start_time,
				}
				metrics.update({k: v.item() for k, v in train_metrics.items()})
				self.logger.log(metrics, 'pretrain')

		self.logger.finish(self.agent)
		print('[Phase 3] Complete. Training finished.')
