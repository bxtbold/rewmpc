import torch
import torch.nn as nn

from networks import init, layers


class FlowPrior(nn.Module):
	"""
	MLP-based conditional rectified flow prior π_θ(τ | c_t).

	Implements Conditional Flow Matching (CFM) with context conditioning via
	concatenation. At inference, uses Euler ODE integration to generate M
	diverse trajectory initializations from noise.

	The conservatism penalty during planning uses the squared L2 distance from
	the flow initialization (a differentiable proxy for -log π_θ), which pulls
	gradient steps back toward the demonstration distribution.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._traj_dim = cfg.horizon * cfg.action_dim

		# Velocity network: maps (τ_s, s, context) → velocity field
		# Input: traj_dim (trajectory) + 1 (flow time) + context_dim
		in_dim = self._traj_dim + 1 + cfg.flow_context_dim
		self._net = layers.mlp(in_dim, 3 * [cfg.flow_mlp_dim], self._traj_dim)
		self.apply(init.weight_init)

	def velocity(self, tau_s, s, context):
		"""
		Predicted velocity field at flow time s.

		Args:
			tau_s:   (B, traj_dim) noisy trajectory at flow time s
			s:       (B, 1) or scalar flow time in [0, 1]
			context: (B, context_dim)

		Returns:
			v: (B, traj_dim) predicted velocity
		"""
		if s.dim() == 0:
			s = s.expand(tau_s.shape[0], 1)
		elif s.dim() == 1:
			s = s.unsqueeze(-1)
		inp = torch.cat([tau_s, s, context], dim=-1)
		return self._net(inp)

	def cfm_loss(self, tau_1, context, weights=None):
		"""
		Conditional Flow Matching loss for a batch of demonstration trajectories.

		Args:
			tau_1:   (B, traj_dim) target trajectories (demo actions, flattened)
			context: (B, context_dim) conditioning context c_t
			weights: (B,) optional return-based importance weights w(τ)

		Returns:
			scalar loss
		"""
		B = tau_1.shape[0]
		tau_0 = torch.randn_like(tau_1)
		s = torch.rand(B, 1, device=tau_1.device)

		# Linear interpolation: τ_s = (1-s)·τ_0 + s·τ_1
		tau_s = (1 - s) * tau_0 + s * tau_1
		target_v = tau_1 - tau_0

		pred_v = self.velocity(tau_s, s, context)
		loss = (pred_v - target_v).pow(2).sum(-1)  # (B,)

		if weights is not None:
			loss = (weights * loss).mean()
		else:
			loss = loss.mean()
		return loss

	@torch.no_grad()
	def sample(self, context, M, nfe=10):
		"""
		Generate M trajectory proposals via Euler ODE integration.

		Args:
			context: (1, context_dim) or (M, context_dim)
			M:       number of parallel proposals
			nfe:     number of function evaluations (Euler steps)

		Returns:
			tau: (M, H, action_dim) action trajectories
		"""
		B = M
		if context.shape[0] == 1:
			context = context.expand(M, -1)

		tau = torch.randn(B, self._traj_dim, device=context.device)
		dt = 1.0 / nfe
		for i in range(nfe):
			s = torch.full((B, 1), i * dt, device=context.device)
			v = self.velocity(tau, s, context)
			tau = tau + dt * v

		tau = tau.clamp(-1., 1.)
		return tau.view(M, self.cfg.horizon, self.cfg.action_dim)

	def log_prior_penalty(self, tau, tau_init):
		"""
		Differentiable conservatism proxy: -log π_θ(τ | c_t) ≈ ||τ - τ_init||².

		This pulls gradient descent back toward the flow prior's support,
		approximating the CQL/IQL pessimism analog at planning time.
		Exact log-likelihood would require Hutchinson trace estimation.

		Args:
			tau:      (M, H, action_dim) current trajectories (leaf, requires_grad)
			tau_init: (M, H, action_dim) detached flow samples (no grad)

		Returns:
			penalty: (M,) scalar per mode
		"""
		diff = (tau - tau_init).pow(2)
		return diff.sum(dim=(-2, -1))  # sum over H and action_dim


def compute_return_weights(returns, alpha):
	"""
	Return-weighted CFM weights: w(τ) = softmax(α · J̃(τ)) over mini-batch.

	Args:
		returns: (B,) trajectory returns (higher is better)
		alpha:   temperature for weighting

	Returns:
		weights: (B,) normalized importance weights
	"""
	normalized = (returns - returns.mean()) / (returns.std() + 1e-8)
	logits = alpha * normalized
	return torch.softmax(logits, dim=0)
