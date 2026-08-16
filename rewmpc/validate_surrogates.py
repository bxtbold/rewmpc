"""
Surrogate validation for ReW-MPC.

Loads a checkpoint and dataset, then runs diagnostics on the reward and value surrogates:
  - Reward: MSE, Pearson correlation, ensemble spread (uncertainty calibration)
  - Value: correlation of V(s_0) with actual episode returns from dataset

Usage:
    python validate_surrogates.py checkpoint=/path/to/phase2.pt
"""
import os
os.environ['MUJOCO_GL'] = 'egl'
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from scipy import stats
from tqdm import tqdm
from copy import deepcopy
from pathlib import Path

import hydra
from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from agent import ReWMPC
from envs import make_env


def _quartile_stats(preds, actuals, label, n_quartiles=4):
    """Print per-quartile prediction quality broken down by actual value."""
    q_edges = np.percentile(actuals, np.linspace(0, 100, n_quartiles + 1))
    print(f'\n  {label} — per-quartile breakdown (by actual reward):')
    print(f'  {"Quartile":<12} {"actual range":>18} {"mean actual":>12} {"mean pred":>10} {"MAE":>8} {"bias":>8} {"corr":>7}')
    print('  ' + '-'*79)
    for q in range(n_quartiles):
        lo, hi = q_edges[q], q_edges[q + 1]
        mask = (actuals >= lo) & (actuals <= hi)
        if mask.sum() < 2:
            continue
        a, p = actuals[mask], preds[mask]
        mae  = np.abs(p - a).mean()
        bias = (p - a).mean()          # positive = over-predicting
        corr = stats.pearsonr(p, a)[0] if a.std() > 1e-6 else float('nan')
        tag = 'Q' + str(q + 1)
        print(f'  {tag:<12} [{lo:6.2f}, {hi:6.2f}]   {a.mean():>10.2f}  {p.mean():>10.2f}  {mae:>8.3f}  {bias:>+8.3f}  {corr:>7.3f}')


@torch.no_grad()
def validate(agent, buffer, cfg, n_batches=200):
    agent.model.eval()
    agent.reward_surrogate.eval()
    agent.value_surrogate.eval()

    reward_preds, reward_actuals = [], []
    v_preds, actual_returns = [], []
    ensemble_stds = []

    for _ in tqdm(range(n_batches), desc='Validating'):
        obs, action, reward, _terminated, task = buffer.sample()

        # Encode all timesteps
        T, B = obs.shape[0], obs.shape[1]
        obs_flat = obs.reshape(T * B, obs.shape[-1])
        task_flat = task.repeat(T) if task is not None else None
        zs = agent.model.encode(obs_flat, task_flat).view(T, B, -1)

        # ---- Reward surrogate ----
        # Compare predicted reward at t=0 vs actual reward at t=0
        r_ensemble = agent.reward_surrogate.forward(zs[0], action[0])  # (K, B, 1)
        r_pred = r_ensemble.mean(0).squeeze(-1)      # (B,)
        r_std  = r_ensemble.std(0).squeeze(-1)        # (B,)
        r_true = reward[0].squeeze(-1)               # (B,)

        reward_preds.append(r_pred.cpu())
        reward_actuals.append(r_true.cpu())
        ensemble_stds.append(r_std.cpu())

        # ---- Value surrogate ----
        # V(s_0) should correlate with sum of actual rewards over horizon
        v0 = agent.value_surrogate.expected(zs[0]).squeeze(-1)  # (B,)
        ep_return = reward.sum(0).squeeze(-1)                    # (B,)

        v_preds.append(v0.cpu())
        actual_returns.append(ep_return.cpu())

    reward_preds   = torch.cat(reward_preds).numpy()
    reward_actuals = torch.cat(reward_actuals).numpy()
    ensemble_stds  = torch.cat(ensemble_stds).numpy()
    v_preds        = torch.cat(v_preds).numpy()
    actual_returns = torch.cat(actual_returns).numpy()

    # ---- Reward metrics ----
    r_mse  = ((reward_preds - reward_actuals) ** 2).mean()
    r_mae  = np.abs(reward_preds - reward_actuals).mean()
    r_corr, r_pval = stats.pearsonr(reward_preds, reward_actuals)
    r_spearman, _  = stats.spearmanr(reward_preds, reward_actuals)
    r_std_mean     = ensemble_stds.mean()
    r_std_std      = ensemble_stds.std()

    # ---- Value metrics ----
    v_corr, v_pval    = stats.pearsonr(v_preds, actual_returns)
    v_spearman, _     = stats.spearmanr(v_preds, actual_returns)
    v_mse             = ((v_preds - actual_returns) ** 2).mean()

    print('\n' + '='*55)
    print('REWARD SURROGATE')
    print('='*55)
    print(f'  MSE:                {r_mse:.4f}')
    print(f'  MAE:                {r_mae:.4f}')
    print(f'  Pearson r:          {r_corr:.4f}  (p={r_pval:.2e})')
    print(f'  Spearman rho:       {r_spearman:.4f}')
    print(f'  Ensemble std mean:  {r_std_mean:.4f}  ± {r_std_std:.4f}')
    print(f'  Actual reward range: [{reward_actuals.min():.2f}, {reward_actuals.max():.2f}]')
    print(f'  Pred   reward range: [{reward_preds.min():.2f}, {reward_preds.max():.2f}]')
    _quartile_stats(reward_preds, reward_actuals, 'Reward')
    print()
    print('='*55)
    print('VALUE SURROGATE  (V(s_0) vs horizon return)')
    print('='*55)
    print(f'  MSE:                {v_mse:.4f}')
    print(f'  Pearson r:          {v_corr:.4f}  (p={v_pval:.2e})')
    print(f'  Spearman rho:       {v_spearman:.4f}')
    print(f'  Actual return range: [{actual_returns.min():.2f}, {actual_returns.max():.2f}]')
    print(f'  Pred  value  range:  [{v_preds.min():.2f}, {v_preds.max():.2f}]')
    _quartile_stats(v_preds, actual_returns, 'Value')
    print('='*55)

    # ---- Interpretation hints ----
    print('\nINTERPRETATION:')
    if r_corr < 0.5:
        print('  [WARN] Reward Pearson r < 0.5 — surrogate is a poor predictor.')
    elif r_corr < 0.8:
        print('  [OK]   Reward Pearson r in 0.5–0.8 — moderate quality.')
    else:
        print('  [GOOD] Reward Pearson r > 0.8 — surrogate is well-fitted.')

    # Low-reward quartile bias check (critical for planning from far states)
    q1_mask = reward_actuals <= np.percentile(reward_actuals, 25)
    q1_bias = (reward_preds[q1_mask] - reward_actuals[q1_mask]).mean()
    if q1_bias > 1.0:
        print(f'  [WARN] Q1 bias = +{q1_bias:.2f} — surrogate OVER-predicts low-reward states.')
        print(f'         Gradient planner sees falsely high reward far from goal → gets stuck.')
    elif q1_bias < -1.0:
        print(f'  [OK]   Q1 bias = {q1_bias:.2f} — surrogate under-predicts low-reward (pessimistic, safer).')
    else:
        print(f'  [OK]   Q1 bias = {q1_bias:+.2f} — low-reward predictions are roughly accurate.')

    if r_std_mean < 1e-3:
        print('  [WARN] Ensemble std ≈ 0 — all heads collapsed to same prediction.')
    else:
        print(f'  [OK]   Ensemble uncertainty is active (mean std={r_std_mean:.4f}).')

    if v_corr < 0.3:
        print('  [WARN] Value Pearson r < 0.3 — poor return prediction.')
    elif v_corr < 0.6:
        print('  [OK]   Value Pearson r in 0.3–0.6 — moderate.')
    else:
        print('  [GOOD] Value Pearson r > 0.6 — good return prediction.')


@hydra.main(config_name='config', config_path='.')
def main(cfg):
    assert torch.cuda.is_available()
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    make_env(cfg)  # populates cfg.obs_shape, cfg.action_dim, cfg.episode_length, etc.

    assert cfg.checkpoint != '???', 'Pass checkpoint=/path/to/phase2.pt'
    agent = ReWMPC(cfg)
    agent.load(cfg.checkpoint)
    print(f'Loaded checkpoint: {cfg.checkpoint}')

    # Load dataset into buffer
    from glob import glob
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
