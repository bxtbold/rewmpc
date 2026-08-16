"""
Flow prior validation for ReW-MPC.

Checks whether the flow prior generates higher-scoring trajectories than random,
whether proposals are diverse, and whether return-weighting is non-degenerate.

Requires a Phase 3 checkpoint (flow prior must be trained).

Usage:
    python validate_flow.py checkpoint=logs/mw-reach/1/single_task/models/final.pt
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from glob import glob

import hydra
from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from agent import ReWMPC
from envs import make_env
from networks.flow_prior import compute_return_weights

M_PROPOSALS = 16


@torch.no_grad()
def score_trajectories(agent, tau, z0, task, cfg):
    """
    Score M trajectories from latent state z0 using surrogate reward rollout.
    Returns (M,) mean reward per step over H steps.

    tau: (M, H, A)
    z0:  (1, D) — encoded obs at t=0
    """
    M, H, A = tau.shape
    z = z0.expand(M, -1)
    h = torch.zeros(M, cfg.latent_dim, device=tau.device)
    total_r = torch.zeros(M, device=tau.device)
    for t in range(H):
        r_pred = agent.reward_surrogate(z, tau[:, t, :]).mean(0).squeeze(-1)  # (M,)
        total_r += r_pred
        z_next = agent.model.next(z, tau[:, t, :], h)
        h = agent.model.gru_step(z, tau[:, t, :], h)
        z = z_next
    return total_r / H


def refine_trajectories(agent, tau_init, z0, h0, task, cfg):
    """
    Run K gradient steps on tau_init minimizing J(tau) = surrogate cost + conservatism.
    Returns refined tau (detached), same shape as tau_init: (M, H, A).
    Model/surrogate parameters are frozen during refinement.
    """
    import torch.nn as nn
    M = tau_init.shape[0]
    z_exp = z0.expand(M, -1)
    h_exp = h0.expand(M, -1)
    discount = float(agent.discount[task].item() if cfg.multitask else agent.discount)

    frozen = (list(agent.model.parameters()) +
              list(agent.reward_surrogate.parameters()) +
              list(agent.value_surrogate.parameters()))
    saved = [p.requires_grad for p in frozen]
    for p in frozen:
        p.requires_grad_(False)

    try:
        with torch.enable_grad():
            tau = tau_init.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([tau], lr=cfg.grad_plan_lr)
            for _ in range(cfg.grad_plan_steps):
                opt.zero_grad()
                J = agent._planning_cost(tau, z_exp, h_exp, tau_init, task, discount)
                J.sum().backward()
                nn.utils.clip_grad_norm_([tau], cfg.grad_clip_norm)
                opt.step()
                with torch.no_grad():
                    tau.data.clamp_(-1., 1.)
    finally:
        for p, req in zip(frozen, saved):
            p.requires_grad_(req)

    return tau.detach()


@torch.no_grad()
def validate(agent, buffer, cfg, n_batches=100):
    agent.model.eval()
    agent.flow_prior.eval()
    agent.reward_surrogate.eval()
    agent.value_surrogate.eval()

    flow_scores, rand_scores = [], []
    refined_flow_scores, refined_rand_scores = [], []
    diversities = []
    all_tau_abs, all_tau_sat = [], []
    weight_spreads = []

    for _ in tqdm(range(n_batches), desc='Validating'):
        obs, action, reward, _terminated, task = buffer.sample()
        B = obs.shape[1]

        obs0 = obs[0]                          # (B, obs_dim)
        task0 = task if task is not None else None
        z0 = agent.model.encode(obs0, task0)  # (B, D)
        h0 = torch.zeros(B, cfg.latent_dim, device=z0.device)
        context = agent.model.context(h0, z0, task0)  # (B, ctx_dim)

        # Score flow proposals vs random for each sample in the batch (first sample only
        # to keep cost reasonable — one context per batch)
        ctx1 = context[:1]                    # (1, ctx_dim)
        z1 = z0[:1]
        h1 = h0[:1]
        task1 = task0[:1] if task0 is not None else None

        tau_flow = agent.flow_prior.sample(ctx1, M=M_PROPOSALS, nfe=cfg.flow_nfe)
        tau_rand = torch.empty_like(tau_flow).uniform_(-1., 1.)

        # Raw (pre-refinement) scores
        flow_s = score_trajectories(agent, tau_flow, z1, task1, cfg).mean().item()
        rand_s = score_trajectories(agent, tau_rand,  z1, task1, cfg).mean().item()
        flow_scores.append(flow_s)
        rand_scores.append(rand_s)

        # Gradient-refined scores — does initialization advantage survive refinement?
        tau_flow_ref = refine_trajectories(agent, tau_flow, z1, h1, task1, cfg)
        tau_rand_ref = refine_trajectories(agent, tau_rand, z1, h1, task1, cfg)
        ref_flow_s = score_trajectories(agent, tau_flow_ref, z1, task1, cfg).mean().item()
        ref_rand_s = score_trajectories(agent, tau_rand_ref, z1, task1, cfg).mean().item()
        refined_flow_scores.append(ref_flow_s)
        refined_rand_scores.append(ref_rand_s)

        # Diversity: mean pairwise L2 / (H * A) over M proposals
        tau_flat = tau_flow.reshape(M_PROPOSALS, -1)               # (M, H*A)
        dists = torch.cdist(tau_flat, tau_flat, p=2)               # (M, M)
        mask = torch.triu(torch.ones(M_PROPOSALS, M_PROPOSALS, dtype=torch.bool,
                                     device=dists.device), diagonal=1)
        div = dists[mask].mean().item() / (cfg.horizon * cfg.action_dim)
        diversities.append(div)

        # Action statistics
        all_tau_abs.append(tau_flow.abs().mean().item())
        all_tau_sat.append((tau_flow.abs() > 0.9).float().mean().item() * 100.)

        # Return-weight spread (using ground-truth episode returns from batch)
        discount = float(agent.discount[task] if cfg.multitask else agent.discount)
        T, B_full = obs.shape[0], obs.shape[1]
        obs_flat = obs.reshape(T * B_full, obs.shape[-1])
        task_flat = task.repeat(T) if task is not None else None
        zs = agent.model.encode(obs_flat, task_flat).view(T, B_full, -1)
        returns = reward.sum(0).squeeze(-1)
        v_terminal = agent.value_surrogate.expected(zs[-1]).squeeze(-1)
        J_returns = returns + (discount ** cfg.horizon) * v_terminal
        weights = compute_return_weights(J_returns, alpha=cfg.flow_alpha_train)
        spread = (weights.max() / (weights.min() + 1e-12)).item()
        weight_spreads.append(min(spread, 1e4))   # cap extreme outliers

    flow_mean  = float(np.mean(flow_scores))
    rand_mean  = float(np.mean(rand_scores))
    improvement = (flow_mean - rand_mean) / (abs(rand_mean) + 1e-8) * 100.

    ref_flow_mean = float(np.mean(refined_flow_scores))
    ref_rand_mean = float(np.mean(refined_rand_scores))
    ref_improvement = (ref_flow_mean - ref_rand_mean) / (abs(ref_rand_mean) + 1e-8) * 100.
    grad_lift_flow = (ref_flow_mean - flow_mean) / (abs(flow_mean) + 1e-8) * 100.
    grad_lift_rand = (ref_rand_mean - rand_mean) / (abs(rand_mean) + 1e-8) * 100.

    diversity    = float(np.mean(diversities))
    mean_abs     = float(np.mean(all_tau_abs))
    pct_sat      = float(np.mean(all_tau_sat))
    weight_spread = float(np.median(weight_spreads))

    print('\n' + '=' * 55)
    print('FLOW PRIOR')
    print('=' * 55)
    print('  Proposal quality — raw (surrogate score, mean reward/step):')
    print(f'    Flow prior:  {flow_mean:+.4f}')
    print(f'    Random:      {rand_mean:+.4f}')
    print(f'    Improvement: {improvement:+.1f}%  (target > +20%)')
    print()
    print(f'  After gradient refinement (K={cfg.grad_plan_steps} steps):')
    print(f'    Flow → refined:   {flow_mean:+.4f}  →  {ref_flow_mean:+.4f}  ({grad_lift_flow:+.1f}%)')
    print(f'    Random → refined: {rand_mean:+.4f}  →  {ref_rand_mean:+.4f}  ({grad_lift_rand:+.1f}%)')
    print(f'    Refined improvement (flow vs rand): {ref_improvement:+.1f}%')
    print()
    print('  Proposal diversity (mean pairwise L2 / dim):')
    print(f'    {diversity:.4f}  (target > 0.15)')
    print()
    print('  Action statistics:')
    print(f'    Mean |action|:  {mean_abs:.3f}  (target < 0.8)')
    print(f'    % saturated:    {pct_sat:.1f}%   (target < 20%)')
    print()
    print('  Return-weight spread (median max/min ratio):')
    print(f'    {weight_spread:.1f}x  (target > 3x)')
    print('=' * 55)

    print('\nINTERPRETATION:')
    if improvement >= 20:
        print(f'  [GOOD] Flow outperforms random by {improvement:.1f}% (raw) — return weighting effective.')
    elif improvement >= 0:
        print(f'  [OK]   Flow marginally better than random ({improvement:.1f}%, raw) — weak signal.')
    else:
        print(f'  [WARN] Flow underperforms random ({improvement:.1f}%, raw) — flow prior may not have converged.')

    if ref_improvement >= 10:
        print(f'  [GOOD] Refined flow still beats refined random by {ref_improvement:.1f}% — '
              f'initialization advantage survives gradient refinement.')
    elif ref_improvement >= 0:
        print(f'  [OK]   Refined flow marginally better than refined random ({ref_improvement:.1f}%) — '
              f'gradient planning largely converges from either init.')
    else:
        print(f'  [WARN] Refined random beats refined flow ({ref_improvement:.1f}%) — '
              f'flow proposals may be landing in worse basins.')

    gap = grad_lift_flow - grad_lift_rand
    if abs(gap) < 5:
        print(f'  [INFO] Gradient lift similar for both inits ({grad_lift_flow:+.1f}% flow, '
              f'{grad_lift_rand:+.1f}% rand) — gradient planning dominates over init quality.')
    elif grad_lift_flow > grad_lift_rand:
        print(f'  [INFO] Flow proposals benefit more from gradient refinement ({grad_lift_flow:+.1f}% '
              f'vs {grad_lift_rand:+.1f}%) — good inits land in better surrogate basins.')
    else:
        print(f'  [INFO] Random proposals benefit more from refinement ({grad_lift_rand:+.1f}% '
              f'vs {grad_lift_flow:+.1f}%) — flow proposals may already be near local optima.')

    if diversity >= 0.15:
        print(f'  [OK]   Diversity {diversity:.3f} — proposals cover action space.')
    else:
        print(f'  [WARN] Diversity {diversity:.3f} < 0.15 — possible mode collapse.')

    if pct_sat < 20:
        print(f'  [OK]   Action saturation {pct_sat:.1f}% — actions not clipped excessively.')
    else:
        print(f'  [WARN] Action saturation {pct_sat:.1f}% — many actions at ±1 boundary.')

    if weight_spread >= 3:
        print(f'  [OK]   Weight spread {weight_spread:.1f}x — return weighting non-degenerate.')
    else:
        print(f'  [WARN] Weight spread {weight_spread:.1f}x < 3x — weights nearly uniform, '
              f'try larger flow_alpha_train.')


@hydra.main(config_name='config', config_path='.')
def main(cfg):
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    make_env(cfg)

    assert cfg.checkpoint != '???', 'Pass checkpoint=/path/to/final.pt'
    agent = ReWMPC(cfg)
    agent.load(cfg.checkpoint)
    print(f'Loaded checkpoint: {cfg.checkpoint}')

    _cfg = deepcopy(cfg)
    _cfg.buffer_size = 50_000_000
    _cfg.steps = _cfg.buffer_size

    if cfg.task.startswith('mw-'):
        fps = [str(Path(cfg.data_dir) / f'{cfg.task}.pt')]
        assert Path(fps[0]).exists(), f'No data at {fps[0]}'
    else:
        fps = sorted(glob(str(Path(cfg.data_dir) / '*.pt')))
        assert fps, f'No data at {cfg.data_dir}'
    buffer = None
    for fp in tqdm(fps, desc='Loading data'):
        td = torch.load(fp, weights_only=False)
        if buffer is None:
            _cfg.episode_length = td.shape[1]
            buffer = Buffer(_cfg)
        buffer.load(td)

    validate(agent, buffer, cfg)


if __name__ == '__main__':
    main()
