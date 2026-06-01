import torch
import torch.nn as nn

from common import math_utils
from networks import init, layers


class RewardSurrogate(nn.Module):
	"""
	Ensemble of MLPs for reward prediction.
	Trained offline on reward-labeled demonstrations with stop-gradient from encoder.
	At planning time, uses LCB: R_lcb = mean - β·std across ensemble heads.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		in_dim = cfg.latent_dim + cfg.action_dim
		self._heads = layers.Ensemble([
			layers.mlp(in_dim, 2 * [cfg.mlp_dim], 1)
			for _ in range(cfg.num_reward_heads)
		])
		self.apply(init.weight_init)

	def forward(self, z, a):
		"""Returns rewards from all heads. Shape: (K, B, 1)."""
		inp = torch.cat([z, a], dim=-1)
		return self._heads(inp)

	def lcb(self, z, a, beta=None):
		"""LCB estimate: mean - β·std across ensemble. Shape: (B, 1)."""
		if beta is None:
			beta = self.cfg.lcb_beta
		rewards = self.forward(z, a)  # (K, B, 1)
		return rewards.mean(0) - beta * rewards.std(0)

	def mse_loss(self, z, a, r_target):
		"""MSE loss on all ensemble heads. z must be stop-gradient from encoder."""
		rewards = self.forward(z.detach(), a)  # (K, B, 1)
		return (rewards - r_target.unsqueeze(0)).pow(2).mean()

	def ranking_loss(self, z_pos, a_pos, z_neg, a_neg, delta=0.1):
		"""
		Hinge ranking loss: penalizes when negative trajectories score higher
		than positive ones (ranked by ground-truth return).
		"""
		r_pos = self.forward(z_pos.detach(), a_pos).mean(0)  # (B, 1)
		r_neg = self.forward(z_neg.detach(), a_neg).mean(0)
		return torch.clamp(r_neg - r_pos + delta, min=0.).mean()

	def bellman_residual(self, z, a, gamma, z_next, value_fn):
		"""
		Bellman residual diagnostic: |R(z,a) + γV(z') - V(z)|.
		High values indicate surrogate extrapolation.
		"""
		with torch.no_grad():
			r = self.forward(z, a).mean(0)
			v = value_fn(z)
			v_next = value_fn(z_next)
			return (r + gamma * v_next - v).abs().mean()


class ValueSurrogate(nn.Module):
	"""
	Ensemble of MLPs predicting terminal value with two-hot symlog targets.
	Trained on N-step TD returns using ground-truth rewards from demonstrations.
	Stop-gradient from encoder and dynamics during training.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._heads = layers.Ensemble([
			layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], cfg.num_bins, dropout=cfg.dropout)
			for _ in range(cfg.num_value_heads)
		])
		self.apply(init.weight_init)

	def forward(self, z):
		"""Returns value logits from all heads. Shape: (K, B, num_bins)."""
		return self._heads(z)

	def expected(self, z):
		"""
		Expected value (scalar) averaged over ensemble heads.
		Uses two-hot inverse to decode from discrete bins. Shape: (B, 1).
		"""
		logits = self.forward(z)  # (K, B, num_bins)
		values = torch.stack([math_utils.two_hot_inv(logits[k], self.cfg) for k in range(self.cfg.num_value_heads)])
		return values.mean(0)

	def td_loss(self, z, td_targets):
		"""
		Soft cross-entropy loss against N-step TD targets.
		z must be stop-gradient from encoder/dynamics.
		"""
		logits = self.forward(z.detach())  # (K, B, num_bins)
		loss = 0.
		for k in range(self.cfg.num_value_heads):
			loss = loss + math_utils.soft_ce(logits[k], td_targets, self.cfg).mean()
		return loss / self.cfg.num_value_heads

	@staticmethod
	def compute_td_targets(rewards, z_final, value_fn, gamma, lambda_td):
		"""
		Compute N-step TD targets from a trajectory segment.

		Args:
			rewards:   (H, B, 1) ground-truth rewards from demonstrations
			z_final:   (B, latent_dim) latent state at end of horizon
			value_fn:  callable (z) → (B, 1) scalar value estimate
			gamma:     (B,) per-sample discounts or scalar
			lambda_td: TD-lambda blending coefficient

		Returns:
			td_target: (B, 1) bootstrap target for each trajectory in batch
		"""
		with torch.no_grad():
			if isinstance(gamma, torch.Tensor) and gamma.dim() == 1:
				gamma = gamma.unsqueeze(-1)  # (B, 1) for broadcasting with (B, 1) targets
			td_target = value_fn(z_final)
			for t in reversed(range(rewards.shape[0])):
				td_target = rewards[t] + gamma * lambda_td * td_target
		return td_target
