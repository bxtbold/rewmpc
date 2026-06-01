"""
Compute per-task reward statistics from offline single-task datasets.

Usage:
    # From local directory
    python datasets/analyze_dataset.py --src datasets/single_task

    # Directly from HuggingFace (no local copy needed)
    python datasets/analyze_dataset.py --hf
    python datasets/analyze_dataset.py --hf --tasks mw-reach mw-drawer-close mw-button-press-topdown
"""
import argparse
from pathlib import Path

import torch

REPO_ID = 'bxtbold/mt10'

MT10_TASKS = [
    'mw-reach', 'mw-push', 'mw-pick-place', 'mw-door-open', 'mw-drawer-open',
    'mw-drawer-close', 'mw-button-press-topdown', 'mw-peg-insert-side',
    'mw-window-open', 'mw-window-close',
]


def analyze(task, path):
    td = torch.load(str(path), weights_only=False)
    reward = td['reward'][:, 1:].float()   # (N_eps, 100) — skip t=0 NaN padding
    n_eps = reward.shape[0]
    r_flat = reward.flatten()
    ep_returns = reward.sum(dim=1)
    return {
        'n_eps':     n_eps,
        'r_mean':    r_flat.mean().item(),
        'r_std':     r_flat.std().item(),
        'r_min':     r_flat.min().item(),
        'r_max':     r_flat.max().item(),
        'ret_mean':  ep_returns.mean().item(),
        'ret_std':   ep_returns.std().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--src', help='Local dir containing per-task subdirs')
    group.add_argument('--hf', action='store_true',
                       help=f'Load directly from HuggingFace ({REPO_ID})')
    parser.add_argument('--tasks', nargs='+', default=MT10_TASKS,
                        choices=MT10_TASKS, metavar='TASK',
                        help='Tasks to analyze (default: all 10)')
    args = parser.parse_args()

    header = (f"{'Task':<30}  {'N_eps':>6}  {'R_mean':>7}  {'R_std':>6}  "
              f"{'R_min':>6}  {'R_max':>6}  {'Ret_mean':>9}  {'Ret_std':>8}")
    print(header)
    print('-' * len(header))

    for task in args.tasks:
        if args.hf:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f'{task}.pt',
                repo_type='dataset',
            )
        else:
            path = Path(args.src) / f'{task}.pt'
            if not path.exists():
                print(f'{task:<30}  NOT FOUND')
                continue

        s = analyze(task, path)
        print(f"{task:<30}  {s['n_eps']:>6}  {s['r_mean']:>7.2f}  {s['r_std']:>6.2f}  "
              f"{s['r_min']:>6.2f}  {s['r_max']:>6.2f}  {s['ret_mean']:>9.1f}  {s['ret_std']:>8.1f}")


if __name__ == '__main__':
    main()
