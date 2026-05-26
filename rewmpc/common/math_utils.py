import torch
import torch.nn.functional as F


def soft_ce(pred, target, cfg):
	"""Cross-entropy loss between predictions and soft two-hot targets (from TD-MPC2)."""
	pred = F.log_softmax(pred, dim=-1)
	target = two_hot(target, cfg)
	return -(target * pred).sum(-1, keepdim=True)


def symlog(x):
	return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
	"""Scalar → soft two-hot encoded targets for discrete regression (from TD-MPC2)."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
	bin_idx = bin_idx.long()
	soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
	return soft_two_hot


def two_hot_inv(x, cfg):
	"""Soft two-hot encoded vector → scalar (from TD-MPC2)."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)
	dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
	return symexp(x)


def sigreg_loss(z):
	"""Regularizes latent distribution toward N(0, I) via mean and variance matching.

	Provides the SigReg effect from the plan: stabilizes latent geometry by
	matching first and second moments of the marginal p(z) to N(0,I).
	"""
	mean_loss = z.mean(0).pow(2).mean()
	var_loss = (z.var(0) - 1.0).pow(2).mean()
	return mean_loss + var_loss


def gaussian_logprob(eps, log_std):
	residual = -0.5 * eps.pow(2) - log_std
	return (residual - 0.9189385175704956).sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
	log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
	return mu, pi, log_pi
