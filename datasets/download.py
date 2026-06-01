"""
Download the MT10 single-task offline dataset from HuggingFace.

The dataset (bxtbold/mt10) contains 10 per-task .pt files, one per MetaWorld
MT10 task, each with ~40k episodes x101 steps.

Usage:
    # Download all 10 tasks
    python datasets/download.py --dst datasets/single_task

    # Download specific tasks only
    python datasets/download.py --dst datasets/single_task \
        --tasks mw-reach mw-drawer-close mw-button-press-topdown
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = 'bxtbold/mt10'

MT10_TASKS = [
    'mw-reach', 'mw-push', 'mw-pick-place', 'mw-door-open', 'mw-drawer-open',
    'mw-drawer-close', 'mw-button-press-topdown', 'mw-peg-insert-side',
    'mw-window-open', 'mw-window-close',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dst', required=True,
                        help='Local destination directory (e.g. datasets/single_task)')
    parser.add_argument('--tasks', nargs='+', default=MT10_TASKS,
                        choices=MT10_TASKS, metavar='TASK',
                        help='Tasks to download (default: all 10)')
    args = parser.parse_args()

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        out_path = dst / f'{task}.pt'

        if out_path.exists():
            print(f'  {task:<35} already exists, skipping.')
            continue

        print(f'  {task:<35} downloading...', end=' ', flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=f'{task}.pt',
            repo_type='dataset',
            local_dir=str(dst),
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f'{size_mb:.0f} MB')

    print(f'\nDone. Data ready in {dst}/')
    print(f'Train with: python train.py task=mw-reach data_dir={dst}')


if __name__ == '__main__':
    main()
