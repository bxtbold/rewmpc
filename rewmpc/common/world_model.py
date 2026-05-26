from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, init


class WorldModel(nn.Module):
	"""
	ReW-MPC world model: MLP encoder with EMA target, GRU recurrent state, and
	JEPA-trained latent dynamics predictor.

	The GRU maintains an episode-level hidden state h_t that summarizes past
	(z, a) pairs. During planning h_t is held fixed; it only advances at real
	environment steps.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.

		# Live encoder (trained via JEPA)
		self._encoder = layers.enc(cfg)

		# EMA target encoder (frozen weights, updated with Polyak averaging)
		self._ema_encoder = deepcopy(self._encoder)
		for p in self._ema_encoder.parameters():
			p.requires_grad_(False)

		# GRU: input = cat(z_t, a_t), hidden = latent_dim
		self._gru = nn.GRU(
			input_size=cfg.latent_dim + cfg.action_dim,
			hidden_size=cfg.latent_dim,
			num_layers=1,
			batch_first=False,
		)

		# Dynamics predictor: f_ψ(z_t, a_t, h_t) → z_{t+1}
		# Input: latent_dim (z) + action_dim (a) + latent_dim (h)
		self._dynamics = layers.mlp(
			cfg.latent_dim + cfg.action_dim + cfg.latent_dim,
			2 * [cfg.mlp_dim],
			cfg.latent_dim,
			act=layers.SimNorm(cfg),
		)

		self.apply(init.weight_init)

	def __repr__(self):
		parts = [
			'ReW-MPC World Model',
			f'Encoder: {self._encoder}',
			f'GRU: {self._gru}',
			f'Dynamics: {self._dynamics}',
			f'Learnable parameters: {self.total_params:,}',
		]
		return '\n'.join(parts)

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def task_emb(self, x, task):
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""Encode observation to latent state (live encoder)."""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		return self._encoder[self.cfg.obs](obs)

	@torch.no_grad()
	def encode_ema(self, obs, task):
		"""Encode observation to latent state (EMA target encoder, stop-gradient)."""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		return self._ema_encoder[self.cfg.obs](obs)

	def next(self, z, a, h):
		"""
		Predict next latent state given (z_t, a_t, h_t).
		Fully differentiable — no stop-gradients — so gradients flow
		through during gradient-based planning.
		"""
		inp = torch.cat([z, a, h], dim=-1)
		return self._dynamics(inp)

	def gru_step(self, z, a, h):
		"""
		Advance GRU hidden state: h_{t+1} = GRU(h_t, cat(z_t, a_t)).
		h: (B, latent_dim); returns h_new: (B, latent_dim).
		"""
		inp = torch.cat([z, a], dim=-1).unsqueeze(0)  # (1, B, input_dim)
		_, h_new = self._gru(inp, h.unsqueeze(0))
		return h_new.squeeze(0)

	def gru_rollout(self, zs, actions, h0=None):
		"""
		Roll out the GRU over a trajectory batch.

		Args:
			zs:      (T, B, latent_dim) — latent states at t=0..T-1
			actions: (T, B, action_dim) — actions at t=0..T-1
			h0:      (B, latent_dim) or None (zero-initialized)

		Returns:
			hs: (T, B, latent_dim) — hidden states h_1..h_T (output AFTER each step)
		"""
		T, B = zs.shape[:2]
		if h0 is None:
			h0 = torch.zeros(B, self.cfg.latent_dim, device=zs.device, dtype=zs.dtype)
		inp = torch.cat([zs, actions], dim=-1)  # (T, B, latent_dim + action_dim)
		hs, _ = self._gru(inp, h0.unsqueeze(0))  # hs: (T, B, latent_dim)
		return hs

	def update_ema(self, ema_tau=0.005):
		"""Polyak update of EMA encoder from live encoder."""
		for p_live, p_ema in zip(self._encoder.parameters(), self._ema_encoder.parameters()):
			p_ema.data.lerp_(p_live.data, ema_tau)

	def context(self, h, z, task=None):
		"""
		Build flow prior context: c_t = cat(h_t, z_t [, task_emb]).
		Returns: (B, flow_context_dim)
		"""
		parts = [h, z]
		if self.cfg.multitask and task is not None:
			if isinstance(task, int):
				task = torch.tensor([task], device=z.device)
			emb = self._task_emb(task.long())
			if emb.shape[0] == 1:
				emb = emb.repeat(z.shape[0], 1)
			parts.append(emb)
		return torch.cat(parts, dim=-1)
