"""
World model validation for ReW-MPC.

Checks JEPA multi-step prediction quality and latent distribution geometry (SigReg health).

Usage:
    python validate_world_model.py checkpoint=logs/mw-reach/1/single_task/models/phase1.pt
    python validate_world_model.py checkpoint=logs/mw-reach/1/single_task/models/phase2.pt
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
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


@torch.no_grad()
def validate(agent, buffer, cfg, n_batches=200):
    agent.model.eval()

    H = cfg.horizon
    cos_sims = [[] for _ in range(H)]
    mses     = [[] for _ in range(H)]
    all_z    = []

    for _ in tqdm(range(n_batches), desc='Validating'):
        obs, action, _reward, _terminated, task = buffer.sample()
        H1, B = obs.shape[0], obs.shape[1]

        obs_flat  = obs.reshape(H1 * B, obs.shape[-1])
        task_flat = task.repeat(H1) if task is not None else None

        zs_live = agent.model.encode(obs_flat, task_flat).view(H1, B, -1)
        zs_ema  = agent.model.encode_ema(
            obs_flat[B:], task_flat[B:] if task_flat is not None else None
        ).view(H, B, -1)

        hs = agent.model.gru_rollout(zs_live[:-1], action)  # (H, B, D)

        for t in range(H):
            h_t = hs[t - 1] if t > 0 else torch.zeros_like(hs[0])
            z_pred = agent.model.next(zs_live[t], action[t], h_t)

            cos_sim = F.cosine_similarity(z_pred, zs_ema[t], dim=-1).mean().item()
            mse     = (z_pred - zs_ema[t]).pow(2).mean().item()
            cos_sims[t].append(cos_sim)
            mses[t].append(mse)

        all_z.append(zs_live.reshape(-1, cfg.latent_dim).cpu())

    # --- Latent distribution (SigReg check) ---
    all_z      = torch.cat(all_z, dim=0)           # (N, D)
    per_dim_mean = all_z.mean(0)                    # (D,)
    per_dim_var  = all_z.var(0)                     # (D,)
    mean_abs     = per_dim_mean.abs().mean().item()
    var_mean     = per_dim_var.mean().item()
    active_pct   = (per_dim_var > 0.1).float().mean().item() * 100.

    cos_means = [float(np.mean(c)) for c in cos_sims]
    mse_means = [float(np.mean(m)) for m in mses]

    print('\n' + '=' * 55)
    print('WORLD MODEL')
    print('=' * 55)
    print('  Cosine similarity (live dynamics vs EMA target):')
    for t, v in enumerate(cos_means):
        bar = '█' * int(v * 20)
        print(f'    t={t+1}:  {v:.4f}  {bar}')
    print('  Prediction MSE per step:')
    mse_str = '  '.join(f't={t+1}: {m:.5f}' for t, m in enumerate(mse_means))
    print(f'    {mse_str}')
    print()
    print('  Latent distribution (SigReg):')
    print(f'    Mean abs:    {mean_abs:.4f}  (target ~0.0)')
    print(f'    Var mean:    {var_mean:.4f}  (target ~1.0)')
    print(f'    Active dims: {active_pct:.1f}%  (target >90%)')
    print('=' * 55)

    print('\nINTERPRETATION:')
    if cos_means[0] >= 0.85:
        print(f'  [GOOD] t=1 cosine sim {cos_means[0]:.3f} >= 0.85 — dynamics well-trained.')
    elif cos_means[0] >= 0.60:
        print(f'  [OK]   t=1 cosine sim {cos_means[0]:.3f} in 0.60–0.85 — moderate.')
    else:
        print(f'  [WARN] t=1 cosine sim {cos_means[0]:.3f} < 0.60 — poor dynamics.')

    if cos_means[-1] >= 0.50:
        print(f'  [OK]   t={H} cosine sim {cos_means[-1]:.3f} >= 0.50 — multi-step degradation acceptable.')
    else:
        print(f'  [WARN] t={H} cosine sim {cos_means[-1]:.3f} < 0.50 — rapid multi-step decay.')

    # SimNorm naturally concentrates some dims; active_pct < 50% is the real warning sign.
    if active_pct >= 50:
        print(f'  [OK]   Active dims {active_pct:.1f}% — no severe collapse.')
    else:
        print(f'  [WARN] Active dims {active_pct:.1f}% < 50% — collapse, check sigreg_coef.')

    # SimNorm outputs live on a simplex (softmax over groups of simnorm_dim dims),
    # so per-dim variance is naturally ~1/simnorm_dim ≈ 0.1 for simnorm_dim=8.
    # The SigReg target of var=1 is inappropriate for SimNorm; check relative spread instead.
    simnorm_expected_var = 1.0 / cfg.simnorm_dim
    if var_mean >= 0.5 * simnorm_expected_var:
        print(f'  [OK]   Var mean {var_mean:.3f} (SimNorm expected ~{simnorm_expected_var:.3f}) — normal.')
    else:
        print(f'  [WARN] Var mean {var_mean:.3f} well below SimNorm expected {simnorm_expected_var:.3f} — possible collapse.')


@hydra.main(config_name='config', config_path='.')
def main(cfg):
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    make_env(cfg)

    assert cfg.checkpoint != '???', 'Pass checkpoint=/path/to/phase1.pt'
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
