"""Smoke tests for rewmpc.

Constructs each network module with a small synthetic config, runs forward
(and where useful, backward) passes on random tensors, and verifies the
result shapes. Everything network-related is CPU-friendly; the optional
end-to-end agent test is skipped unless CUDA is available (the agent
hardcodes `cuda:0`).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

# rewmpc uses absolute imports (`from common import ...`, `from networks import ...`)
# which only resolve when rewmpc/rewmpc/ is on sys.path; mirror what train.py expects.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'rewmpc'))

from common import math_utils  # noqa: E402
from networks.flow_prior import FlowPrior, compute_return_weights  # noqa: E402
from networks.layers import (  # noqa: E402
	Ensemble,
	SimNorm,
	enc,
	mlp,
	spectral_norm_mlp,
)
from networks.surrogates import RewardSurrogate, ValueSurrogate  # noqa: E402
from networks.world_model import WorldModel  # noqa: E402


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
HAS_NN_BUFFER = hasattr(torch.nn, 'Buffer')


def make_cfg() -> SimpleNamespace:
	"""Tiny CPU-friendly config that satisfies every module's attribute reads."""
	cfg = SimpleNamespace(
		# observation / action
		obs='state',
		obs_shape={'state': (24,)},
		action_dim=6,
		# multitask off for smoke
		multitask=False,
		tasks=['task0'],
		task_dim=0,
		action_dims=[6],
		episode_length=100,
		episode_lengths=[100],
		# representation
		latent_dim=64,
		enc_dim=64,
		mlp_dim=128,
		num_enc_layers=2,
		simnorm_dim=8,
		# rollouts
		horizon=4,
		rho=0.5,
		# surrogates
		num_reward_heads=3,
		num_value_heads=3,
		num_bins=21,
		vmin=-10.0,
		vmax=10.0,
		bin_size=20.0 / 20,
		dropout=0.0,
		lcb_beta=1.0,
		rank_loss_coef=0.1,
		td_lambda=0.95,
		# flow prior
		flow_mlp_dim=64,
		flow_modes=4,
		flow_nfe=5,
		flow_alpha=0.5,
		flow_alpha_train=2.0,
		# planning
		grad_plan_lr=0.05,
		grad_plan_steps=2,
		grad_clip_norm=20.0,
		plan_noise_std=0.0,
		action_reg_u=0.01,
		action_reg_delta=0.01,
		# training
		lr=3e-4,
		flow_lr=1e-4,
		enc_lr_scale=0.3,
		jepa_coef=1.0,
		sigreg_coef=0.1,
		ema_tau=0.005,
		# discount
		discount_denom=5,
		discount_min=0.95,
		discount_max=0.995,
	)
	cfg.flow_traj_dim = cfg.horizon * cfg.action_dim
	cfg.flow_context_dim = 2 * cfg.latent_dim + cfg.task_dim
	return cfg


class Skip(Exception):
	"""Raised by a test to indicate it was intentionally skipped."""


PASS = 0
FAIL = 0
SKIP = 0


def run(name: str, fn) -> None:
	global PASS, FAIL, SKIP
	try:
		fn()
		PASS += 1
		print(f'  PASS  {name}')
	except Skip as e:
		SKIP += 1
		print(f'  SKIP  {name}: {e}')
	except Exception as e:
		FAIL += 1
		print(f'  FAIL  {name}: {type(e).__name__}: {e}')


# ----------------------------- layers -----------------------------

def test_mlp() -> None:
	m = mlp(8, [16, 16], 4).to(DEVICE)
	y = m(torch.randn(3, 8, device=DEVICE))
	assert y.shape == (3, 4)


def test_simnorm() -> None:
	cfg = make_cfg()
	y = SimNorm(cfg)(torch.randn(3, 16, device=DEVICE))
	assert y.shape == (3, 16)
	# SimNorm groups must sum to 1 along the last softmax axis.
	groups = y.view(3, -1, cfg.simnorm_dim)
	assert torch.allclose(groups.sum(-1), torch.ones(3, groups.shape[1], device=DEVICE), atol=1e-5)


def test_ensemble() -> None:
	models = [mlp(8, [16], 4) for _ in range(3)]
	ens = Ensemble(models).to(DEVICE)
	y = ens(torch.randn(5, 8, device=DEVICE))
	assert y.shape == (3, 5, 4)


def test_spectral_norm_mlp() -> None:
	m = spectral_norm_mlp(8, [16, 16], 4).to(DEVICE)
	y = m(torch.randn(2, 8, device=DEVICE))
	assert y.shape == (2, 4)


def test_enc() -> None:
	cfg = make_cfg()
	e = enc(cfg).to(DEVICE)
	y = e['state'](torch.randn(3, cfg.obs_shape['state'][0], device=DEVICE))
	assert y.shape == (3, cfg.latent_dim)


# --------------------------- math_utils ---------------------------

def test_symlog_symexp_roundtrip() -> None:
	x = torch.randn(8) * 5
	assert torch.allclose(math_utils.symexp(math_utils.symlog(x)), x, atol=1e-5)


def test_two_hot() -> None:
	cfg = make_cfg()
	x = torch.randn(8, 1) * 5
	th = math_utils.two_hot(x, cfg)
	assert th.shape == (8, cfg.num_bins)
	# soft two-hot rows should sum to 1
	assert torch.allclose(th.sum(-1), torch.ones(8), atol=1e-5)


def test_sigreg_loss() -> None:
	loss = math_utils.sigreg_loss(torch.randn(64, 32))
	assert loss.numel() == 1 and torch.isfinite(loss)


def test_soft_ce() -> None:
	cfg = make_cfg()
	pred = torch.randn(4, cfg.num_bins, requires_grad=True)
	target = torch.randn(4, 1)
	loss = math_utils.soft_ce(pred, target, cfg).mean()
	loss.backward()
	assert pred.grad is not None and torch.isfinite(loss)


# --------------------------- world model --------------------------

def test_world_model_forward() -> None:
	cfg = make_cfg()
	m = WorldModel(cfg).to(DEVICE)
	B = 3
	obs = torch.randn(B, cfg.obs_shape['state'][0], device=DEVICE)
	a = torch.randn(B, cfg.action_dim, device=DEVICE)
	h = torch.zeros(B, cfg.latent_dim, device=DEVICE)
	z = m.encode(obs, task=None)
	assert z.shape == (B, cfg.latent_dim)
	z_next = m.next(z, a, h)
	assert z_next.shape == (B, cfg.latent_dim)
	h_new = m.gru_step(z, a, h)
	assert h_new.shape == (B, cfg.latent_dim)


def test_world_model_gru_rollout() -> None:
	cfg = make_cfg()
	m = WorldModel(cfg).to(DEVICE)
	T, B = cfg.horizon, 3
	zs = torch.randn(T, B, cfg.latent_dim, device=DEVICE)
	a = torch.randn(T, B, cfg.action_dim, device=DEVICE)
	hs = m.gru_rollout(zs, a)
	assert hs.shape == (T, B, cfg.latent_dim)


def test_world_model_context() -> None:
	cfg = make_cfg()
	m = WorldModel(cfg).to(DEVICE)
	B = 3
	c = m.context(
		torch.zeros(B, cfg.latent_dim, device=DEVICE),
		torch.zeros(B, cfg.latent_dim, device=DEVICE),
	)
	assert c.shape == (B, cfg.flow_context_dim)


def test_world_model_ema_update() -> None:
	cfg = make_cfg()
	m = WorldModel(cfg).to(DEVICE)
	# Make encoder weights diverge from the EMA copy, then take an EMA step.
	with torch.no_grad():
		for p in m._encoder.parameters():
			p.add_(torch.randn_like(p))
	before = [p.clone() for p in m._ema_encoder.parameters()]
	m.update_ema(ema_tau=0.5)
	after = list(m._ema_encoder.parameters())
	assert any(not torch.equal(b, a) for b, a in zip(before, after))


# ---------------------------- surrogates --------------------------

def test_reward_surrogate() -> None:
	cfg = make_cfg()
	R = RewardSurrogate(cfg).to(DEVICE)
	B = 5
	z = torch.randn(B, cfg.latent_dim, device=DEVICE)
	a = torch.randn(B, cfg.action_dim, device=DEVICE)
	r = R(z, a)
	assert r.shape == (cfg.num_reward_heads, B, 1)
	assert R.lcb(z, a).shape == (B, 1)
	loss = R.mse_loss(z, a, torch.randn(B, 1, device=DEVICE))
	loss.backward()
	assert any(p.grad is not None for p in R.parameters())


def test_reward_surrogate_ranking() -> None:
	cfg = make_cfg()
	R = RewardSurrogate(cfg).to(DEVICE)
	B = 4
	z = torch.randn(B, cfg.latent_dim, device=DEVICE)
	a = torch.randn(B, cfg.action_dim, device=DEVICE)
	loss = R.ranking_loss(z[:2], a[:2], z[2:], a[2:])
	assert loss.numel() == 1 and torch.isfinite(loss)


def test_value_surrogate() -> None:
	cfg = make_cfg()
	V = ValueSurrogate(cfg).to(DEVICE)
	B = 5
	z = torch.randn(B, cfg.latent_dim, device=DEVICE)
	assert V(z).shape == (cfg.num_value_heads, B, cfg.num_bins)
	assert V.expected(z).shape == (B, 1)


def test_value_td_loss() -> None:
	cfg = make_cfg()
	V = ValueSurrogate(cfg).to(DEVICE)
	B = 5
	z = torch.randn(B, cfg.latent_dim, device=DEVICE)
	td_targets = torch.randn(B, 1, device=DEVICE)
	loss = V.td_loss(z, td_targets)
	loss.backward()
	assert torch.isfinite(loss) and any(p.grad is not None for p in V.parameters())


def test_compute_td_targets() -> None:
	cfg = make_cfg()
	V = ValueSurrogate(cfg).to(DEVICE)
	B, H = 4, cfg.horizon
	rewards = torch.randn(H, B, 1, device=DEVICE)
	z_final = torch.randn(B, cfg.latent_dim, device=DEVICE)
	td = ValueSurrogate.compute_td_targets(rewards, z_final, V.expected, gamma=0.99, lambda_td=0.95)
	assert td.shape == (B, 1) and torch.isfinite(td).all()


# --------------------------- flow prior ---------------------------

def test_flow_prior_cfm_loss() -> None:
	cfg = make_cfg()
	F = FlowPrior(cfg).to(DEVICE)
	B = 5
	tau_1 = torch.randn(B, cfg.flow_traj_dim, device=DEVICE)
	ctx = torch.randn(B, cfg.flow_context_dim, device=DEVICE)
	loss = F.cfm_loss(tau_1, ctx)
	loss.backward()
	assert loss.numel() == 1 and torch.isfinite(loss)
	assert any(p.grad is not None for p in F.parameters())


def test_flow_prior_cfm_loss_weighted() -> None:
	cfg = make_cfg()
	F = FlowPrior(cfg).to(DEVICE)
	B = 5
	tau_1 = torch.randn(B, cfg.flow_traj_dim, device=DEVICE)
	ctx = torch.randn(B, cfg.flow_context_dim, device=DEVICE)
	weights = torch.softmax(torch.randn(B, device=DEVICE), dim=0)
	loss = F.cfm_loss(tau_1, ctx, weights=weights)
	loss.backward()
	assert loss.numel() == 1 and torch.isfinite(loss)


def test_flow_prior_sample() -> None:
	cfg = make_cfg()
	F = FlowPrior(cfg).to(DEVICE)
	ctx = torch.randn(1, cfg.flow_context_dim, device=DEVICE)
	tau = F.sample(ctx, M=cfg.flow_modes, nfe=cfg.flow_nfe)
	assert tau.shape == (cfg.flow_modes, cfg.horizon, cfg.action_dim)
	assert (tau >= -1).all() and (tau <= 1).all()


def test_log_prior_penalty() -> None:
	cfg = make_cfg()
	F = FlowPrior(cfg).to(DEVICE)
	tau = torch.randn(cfg.flow_modes, cfg.horizon, cfg.action_dim, device=DEVICE, requires_grad=True)
	tau_init = torch.randn(cfg.flow_modes, cfg.horizon, cfg.action_dim, device=DEVICE)
	p = F.log_prior_penalty(tau, tau_init)
	assert p.shape == (cfg.flow_modes,)
	p.sum().backward()
	assert tau.grad is not None


def test_compute_return_weights() -> None:
	w = compute_return_weights(torch.randn(16), alpha=2.0)
	assert w.shape == (16,)
	assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)


# --------------------------- agent (e2e) --------------------------

def test_agent_end_to_end() -> None:
	"""Build agent, run one step of each phase + planning + save/load."""
	if not torch.cuda.is_available():
		raise Skip('requires CUDA (agent.py hardcodes cuda:0)')
	if not HAS_NN_BUFFER:
		raise Skip('requires torch>=2.5 (torch.nn.Buffer)')

	from agent import ReWMPC  # noqa: WPS433 — deferred to keep CPU-only path import-safe

	cfg = make_cfg()
	agent = ReWMPC(cfg)
	B, H, O = 4, cfg.horizon, cfg.obs_shape['state'][0]

	# agent.update_* paths don't auto-move inputs; build directly on agent.device.
	obs = torch.randn(H + 1, B, O, device=agent.device)
	a = torch.randn(H, B, cfg.action_dim, device=agent.device)
	r = torch.randn(H, B, 1, device=agent.device)

	out = agent.update_world(obs, a, task=None)
	assert 'world/jepa_loss' in out

	out = agent.update_surrogates(obs, a, r, task=None)
	assert 'surrogate/reward_loss' in out and 'surrogate/value_loss' in out

	out = agent.update_flow(obs, a, r, task=None, use_weights=False)
	assert 'flow/cfm_loss' in out

	action = agent.act(obs[0, 0], t0=True, eval_mode=True, task=None)
	assert action.shape == (cfg.action_dim,)

	with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
		agent.save(f.name)
		agent.load(f.name)


# ------------------------------- main -----------------------------

SECTIONS = [
	('layers', [
		('mlp', test_mlp),
		('SimNorm', test_simnorm),
		('Ensemble', test_ensemble),
		('spectral_norm_mlp', test_spectral_norm_mlp),
		('enc (state)', test_enc),
	]),
	('math_utils', [
		('symlog/symexp roundtrip', test_symlog_symexp_roundtrip),
		('two_hot rows sum to 1', test_two_hot),
		('sigreg_loss', test_sigreg_loss),
		('soft_ce backward', test_soft_ce),
	]),
	('world_model', [
		('encode + next + gru_step shapes', test_world_model_forward),
		('gru_rollout shapes', test_world_model_gru_rollout),
		('context shape', test_world_model_context),
		('EMA update moves params', test_world_model_ema_update),
	]),
	('surrogates', [
		('RewardSurrogate forward + lcb + mse backward', test_reward_surrogate),
		('RewardSurrogate ranking_loss', test_reward_surrogate_ranking),
		('ValueSurrogate forward + expected', test_value_surrogate),
		('ValueSurrogate td_loss backward', test_value_td_loss),
		('compute_td_targets', test_compute_td_targets),
	]),
	('flow_prior', [
		('cfm_loss (unweighted) backward', test_flow_prior_cfm_loss),
		('cfm_loss (weighted) backward', test_flow_prior_cfm_loss_weighted),
		('sample shape + range', test_flow_prior_sample),
		('log_prior_penalty backward through tau', test_log_prior_penalty),
		('compute_return_weights sums to 1', test_compute_return_weights),
	]),
	('agent (end-to-end)', [
		('ReWMPC three-phase update + act + save/load', test_agent_end_to_end),
	]),
]


def main() -> int:
	torch.manual_seed(0)
	print(f'rewmpc smoke tests (device={DEVICE}, torch={torch.__version__})')
	print('-' * 64)
	for section_name, cases in SECTIONS:
		print(f'[{section_name}]')
		for name, fn in cases:
			run(name, fn)
	print('-' * 64)
	print(f'PASS {PASS}   FAIL {FAIL}   SKIP {SKIP}')
	return 0 if FAIL == 0 else 1


if __name__ == '__main__':
	sys.exit(main())

