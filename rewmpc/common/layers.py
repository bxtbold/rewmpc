import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from tensordict import from_modules


class Ensemble(nn.Module):
	"""Vectorized ensemble of modules (from TD-MPC2)."""

	def __init__(self, modules, **kwargs):
		super().__init__()
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class SimNorm(nn.Module):
	"""Simplicial normalization (from TD-MPC2)."""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""Linear layer with LayerNorm, activation, and optional dropout (from TD-MPC2)."""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return (f"NormedLinear(in_features={self.in_features}, "
				f"out_features={self.out_features}, "
				f"bias={self.bias is not None}{repr_dropout}, "
				f"act={self.act.__class__.__name__})")


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""MLP with LayerNorm and Mish activations (from TD-MPC2)."""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	layers = nn.ModuleList()
	for i in range(len(dims) - 2):
		layers.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*layers)


def enc(cfg, out={}):
	"""Encoder factory for state observations (from TD-MPC2)."""
	for k in cfg.obs_shape.keys():
		if k == 'state':
			out[k] = mlp(cfg.obs_shape[k][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented in rewmpc.")
	return nn.ModuleDict(out)


def spectral_norm_mlp(in_dim, mlp_dims, out_dim, dropout=0.):
	"""MLP with spectral normalization on all linear layers (for reward surrogate)."""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	layers = nn.ModuleList()
	for i in range(len(dims) - 2):
		layer_list = []
		if dropout and i == 0:
			layer_list.append(nn.Dropout(dropout))
		layer_list.append(nn.utils.spectral_norm(nn.Linear(dims[i], dims[i+1])))
		layer_list.append(nn.LayerNorm(dims[i+1]))
		layer_list.append(nn.Mish(inplace=False))
		layers.append(nn.Sequential(*layer_list))
	layers.append(nn.utils.spectral_norm(nn.Linear(dims[-2], dims[-1])))
	return nn.Sequential(*layers)
