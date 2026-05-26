import torch
import torch.nn as nn
from tensordict import TensorDict

from common import math_utils
from networks.world_model import WorldModel
from networks.surrogates import RewardSurrogate, ValueSurrogate
from networks.flow_prior import FlowPrior, compute_return_weights


class ReWMPC(nn.Module):
	"""
	ReW-MPC agent. Implements:
	  - Phase 1: world model training (JEPA + SigReg)
	  - Phase 2: reward & value surrogate training (offline, stop-grad from encoder)
	  - Phase 3: return-weighted flow prior training
	  - Online control: gradient-based MPC with flow prior initialization
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')

		self.model = WorldModel(cfg).to(self.device)
		self.reward_surrogate = RewardSurrogate(cfg).to(self.device)
		self.value_surrogate = ValueSurrogate(cfg).to(self.device)
		self.flow_prior = FlowPrior(cfg).to(self.device)

		# Phase 1: encoder + GRU + dynamics
		self.world_optim = torch.optim.AdamW([
			{'params': self.model._encoder.parameters(), 'lr': cfg.lr * cfg.enc_lr_scale},
			{'params': self.model._gru.parameters()},
			{'params': self.model._dynamics.parameters()},
			*([{'params': self.model._task_emb.parameters()}] if cfg.multitask else []),
		], lr=cfg.lr, weight_decay=1e-4)

		# Phase 2: reward and value surrogates
		self.reward_optim = torch.optim.AdamW(self.reward_surrogate.parameters(), lr=cfg.lr)
		self.value_optim = torch.optim.AdamW(self.value_surrogate.parameters(), lr=cfg.lr)

		# Phase 3: flow prior
		self.flow_optim = torch.optim.AdamW(self.flow_prior.parameters(), lr=cfg.flow_lr)

		self.discount = (
			torch.tensor([self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device)
			if cfg.multitask else self._get_discount(cfg.episode_length)
		)

		# Per-episode GRU hidden state, updated in act()
		self._h = torch.nn.Buffer(torch.zeros(1, cfg.latent_dim, device=self.device))

		self.model.eval()
		self.reward_surrogate.eval()
		self.value_surrogate.eval()
		self.flow_prior.eval()

	def _get_discount(self, episode_length):
		frac = episode_length / self.cfg.discount_denom
		return min(max((frac - 1) / frac, self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		torch.save({
			'model': self.model.state_dict(),
			'reward_surrogate': self.reward_surrogate.state_dict(),
			'value_surrogate': self.value_surrogate.state_dict(),
			'flow_prior': self.flow_prior.state_dict(),
		}, fp)

	def load(self, fp):
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=self.device, weights_only=False)
		self.model.load_state_dict(state_dict['model'])
		if 'reward_surrogate' in state_dict:
			self.reward_surrogate.load_state_dict(state_dict['reward_surrogate'])
		if 'value_surrogate' in state_dict:
			self.value_surrogate.load_state_dict(state_dict['value_surrogate'])
		if 'flow_prior' in state_dict:
			self.flow_prior.load_state_dict(state_dict['flow_prior'])

	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action via gradient-based planning.

		NOTE: no @torch.no_grad() here — planning requires gradient computation.
		Encoding and GRU update are done inside no_grad blocks.
		"""
		with torch.no_grad():
			obs = obs.to(self.device, non_blocking=True)
			if obs.dim() == 1:
				obs = obs.unsqueeze(0)
			if task is not None:
				task = torch.tensor([task], device=self.device)
			if t0:
				self._h.zero_()
			z = self.model.encode(obs, task).detach()
			h = self._h.clone().detach()

		action = self._plan(z, h, task, eval_mode)

		with torch.no_grad():
			self._h.copy_(self.model.gru_step(z, action, self._h))

		return action.squeeze(0).cpu()

	def _plan(self, z_t, h_t, task, eval_mode=False):
		"""
		Gradient-based MPC: sample M flow prior proposals then refine with K
		gradient steps on J(τ) using surrogate reward, value, and conservatism penalty.

		Model parameters are temporarily frozen to avoid polluting their gradients.
		"""
		cfg = self.cfg

		with torch.no_grad():
			context = self.model.context(h_t, z_t, task)
			tau_init = self.flow_prior.sample(context, M=cfg.flow_modes, nfe=cfg.flow_nfe)

		# Freeze model parameters during planning so backward only touches tau
		frozen_params = (
			list(self.model.parameters()) +
			list(self.reward_surrogate.parameters()) +
			list(self.value_surrogate.parameters())
		)
		saved_req_grad = [p.requires_grad for p in frozen_params]
		for p in frozen_params:
			p.requires_grad_(False)

		try:
			with torch.enable_grad():
				tau = tau_init.detach().clone().requires_grad_(True)
				optimizer = torch.optim.Adam([tau], lr=cfg.grad_plan_lr)

				z_expand = z_t.expand(cfg.flow_modes, -1)  # (M, latent_dim)
				h_expand = h_t.expand(cfg.flow_modes, -1)  # (M, latent_dim)
				discount = self.discount[task].item() if cfg.multitask else float(self.discount)

				for _ in range(cfg.grad_plan_steps):
					optimizer.zero_grad()
					J = self._planning_cost(tau, z_expand, h_expand, tau_init, task, discount)
					J.sum().backward()
					nn.utils.clip_grad_norm_([tau], cfg.grad_clip_norm)
					optimizer.step()
					with torch.no_grad():
						tau.data.clamp_(-1., 1.)

			with torch.no_grad():
				final_J = self._planning_cost(
					tau.detach(), z_expand, h_expand, tau_init, task, discount
				)
		finally:
			for p, req in zip(frozen_params, saved_req_grad):
				p.requires_grad_(req)

		best = final_J.argmin()
		a = tau[best, 0].detach()  # (action_dim,)

		if not eval_mode:
			a = a + cfg.plan_noise_std * torch.randn_like(a)
			a = a.clamp(-1., 1.)

		return a.unsqueeze(0)  # (1, action_dim)

	def _planning_cost(self, tau, z_t, h_t, tau_init, task, discount):
		"""
		J(τ) for a batch of M trajectories.

		  J = Σ_k [ -R_lcb(ẑ_k, a_k) + reg(a_k) ] - γ^H V(ẑ_H) + α ||τ - τ_init||²

		Args:
			tau:      (M, H, action_dim) — the trajectories being optimized
			z_t:      (M, latent_dim)    — current latent (fixed, no grad)
			h_t:      (M, latent_dim)    — GRU hidden state (fixed, no grad)
			tau_init: (M, H, action_dim) — detached flow proposals (no grad)
			task:     task index or None
			discount: scalar γ

		Returns:
			J: (M,) total cost per mode
		"""
		cfg = self.cfg
		M, H, _ = tau.shape

		J = torch.zeros(M, device=tau.device)
		z = z_t
		discount_k = 1.0

		for k in range(H):
			a_k = tau[:, k]  # (M, action_dim)
			prev_a = tau[:, k - 1] if k > 0 else torch.zeros_like(a_k)

			r_lcb = self.reward_surrogate.lcb(z, a_k, beta=cfg.lcb_beta).squeeze(-1)  # (M,)
			reg = (cfg.action_reg_u * a_k.pow(2).sum(-1) +
				   cfg.action_reg_delta * (a_k - prev_a).pow(2).sum(-1))

			J = J + discount_k * (-r_lcb + reg)
			discount_k = discount_k * discount

			z = self.model.next(z, a_k, h_t)  # differentiable rollout

		v_terminal = self.value_surrogate.expected(z).squeeze(-1)  # (M,)
		J = J - discount_k * v_terminal

		# Conservatism: squared deviation from flow initialization
		J = J + cfg.flow_alpha * self.flow_prior.log_prior_penalty(tau, tau_init)

		return J

	# ------------------------------------------------------------------
	# Phase 1: World model (JEPA + SigReg)
	# ------------------------------------------------------------------

	def update_world(self, obs, action, task=None):
		"""
		Trains encoder + GRU + dynamics via JEPA prediction loss and SigReg.

		JEPA: predict next latent (f_ψ(z_t, a_t, h_t)) against EMA-encoder target.
		SigReg: regularize latent marginal toward N(0, I).
		"""
		self.model.train()
		H = action.shape[0]

		# Live latents (gradient flows through encoder)
		zs_live = torch.stack([self.model.encode(obs[t], task) for t in range(H + 1)])  # (H+1, B, D)

		# EMA-target latents (stop-gradient)
		zs_ema = torch.stack([self.model.encode_ema(obs[t], task) for t in range(1, H + 1)])  # (H, B, D)

		# GRU hidden states h_1..h_H from (z_0..z_{H-1}, a_0..a_{H-1})
		hs = self.model.gru_rollout(zs_live[:-1], action)  # (H, B, D)

		# Predict next latents: f_ψ(z_t, a_t, h_t)
		jepa_loss = 0.
		for t in range(H):
			h_t = hs[t - 1] if t > 0 else torch.zeros_like(hs[0])
			z_pred = self.model.next(zs_live[t], action[t], h_t)
			jepa_loss = jepa_loss + (z_pred - zs_ema[t]).pow(2).mean() * self.cfg.rho ** t
		jepa_loss = jepa_loss / H

		# SigReg: match latent marginal to N(0, I)
		all_z = zs_live.reshape(-1, self.cfg.latent_dim)
		sigreg = math_utils.sigreg_loss(all_z)

		total = self.cfg.jepa_coef * jepa_loss + self.cfg.sigreg_coef * sigreg

		self.world_optim.zero_grad(set_to_none=True)
		total.backward()
		grad_norm = nn.utils.clip_grad_norm_(
			list(self.model._encoder.parameters()) +
			list(self.model._gru.parameters()) +
			list(self.model._dynamics.parameters()) +
			(list(self.model._task_emb.parameters()) if self.cfg.multitask else []),
			self.cfg.grad_clip_norm,
		)
		self.world_optim.step()
		self.model.update_ema(ema_tau=self.cfg.ema_tau)
		self.model.eval()

		return TensorDict({
			'world/jepa_loss': jepa_loss.detach(),
			'world/sigreg_loss': sigreg.detach(),
			'world/total_loss': total.detach(),
			'world/grad_norm': grad_norm,
		})

	# ------------------------------------------------------------------
	# Phase 2: Reward + Value surrogates
	# ------------------------------------------------------------------

	def update_surrogates(self, obs, action, reward, task=None):
		"""
		Trains reward and value surrogates with stop-gradient from the encoder.
		Reward: MSE + optional hinge ranking loss.
		Value: soft cross-entropy against N-step TD targets.
		"""
		self.reward_surrogate.train()
		self.value_surrogate.train()

		with torch.no_grad():
			zs = torch.stack([self.model.encode(obs[t], task) for t in range(obs.shape[0])])

		H = action.shape[0]
		B = obs.shape[1]

		# --- Reward surrogate ---
		reward_loss = 0.
		for t in range(H):
			reward_loss = reward_loss + self.reward_surrogate.mse_loss(zs[t], action[t], reward[t])
		reward_loss = reward_loss / H

		if self.cfg.rank_loss_coef > 0 and B >= 4:
			returns = reward.sum(0).squeeze(-1)  # (B,)
			sorted_idx = returns.argsort()
			half = B // 2
			rank_loss = self.reward_surrogate.ranking_loss(
				zs[0][sorted_idx[half:]], action[0][sorted_idx[half:]],
				zs[0][sorted_idx[:half]], action[0][sorted_idx[:half]],
			)
			reward_loss = reward_loss + self.cfg.rank_loss_coef * rank_loss

		self.reward_optim.zero_grad(set_to_none=True)
		reward_loss.backward()
		nn.utils.clip_grad_norm_(self.reward_surrogate.parameters(), self.cfg.grad_clip_norm)
		self.reward_optim.step()

		# --- Value surrogate ---
		discount = self.discount[task] if self.cfg.multitask else self.discount  # (B,) or scalar
		td_targets = ValueSurrogate.compute_td_targets(
			rewards=reward,
			z_final=zs[-1],
			value_fn=self.value_surrogate.expected,
			gamma=discount,
			lambda_td=self.cfg.td_lambda,
		)
		value_loss = self.value_surrogate.td_loss(zs[0], td_targets)

		self.value_optim.zero_grad(set_to_none=True)
		value_loss.backward()
		nn.utils.clip_grad_norm_(self.value_surrogate.parameters(), self.cfg.grad_clip_norm)
		self.value_optim.step()

		self.reward_surrogate.eval()
		self.value_surrogate.eval()

		return TensorDict({
			'surrogate/reward_loss': reward_loss.detach(),
			'surrogate/value_loss': value_loss.detach(),
		})

	# ------------------------------------------------------------------
	# Phase 3: Flow prior
	# ------------------------------------------------------------------

	def update_flow(self, obs, action, reward, task=None, use_weights=True):
		"""
		Return-weighted CFM training of the flow prior.
		During warm-up (use_weights=False) trains with uniform weights (pure BC).
		"""
		self.flow_prior.train()

		with torch.no_grad():
			zs = torch.stack([self.model.encode(obs[t], task) for t in range(obs.shape[0])])
			h0 = torch.zeros(obs.shape[1], self.cfg.latent_dim, device=self.device)
			context = self.model.context(h0, zs[0], task)

		B = action.shape[1]
		# Demonstration trajectories as target: flatten (H, B, A) → (B, H*A)
		tau_1 = action.permute(1, 0, 2).reshape(B, -1)

		with torch.no_grad():
			discount = self.discount[task] if self.cfg.multitask else self.discount  # (B,) or scalar
			returns = reward.sum(0).squeeze(-1)
			v_terminal = self.value_surrogate.expected(zs[-1]).squeeze(-1)
			J_returns = returns + (discount ** self.cfg.horizon) * v_terminal

		weights = compute_return_weights(J_returns, alpha=self.cfg.flow_alpha_train) if use_weights else None
		flow_loss = self.flow_prior.cfm_loss(tau_1, context, weights=weights)

		self.flow_optim.zero_grad(set_to_none=True)
		flow_loss.backward()
		nn.utils.clip_grad_norm_(self.flow_prior.parameters(), self.cfg.grad_clip_norm)
		self.flow_optim.step()
		self.flow_prior.eval()

		return TensorDict({
			'flow/cfm_loss': flow_loss.detach(),
			'flow/return_mean': J_returns.mean().detach(),
		})
