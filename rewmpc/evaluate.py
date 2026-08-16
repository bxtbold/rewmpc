import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')

from glob import glob
from pathlib import Path

import hydra
import numpy as np
import torch
from termcolor import colored

from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from agent import ReWMPC

torch.backends.cudnn.benchmark = True


def find_checkpoint(cfg):
    """Auto-find the latest checkpoint for this task/seed/exp_name."""
    models_dir = Path(cfg.work_dir) / 'models'
    if not models_dir.exists():
        raise FileNotFoundError(f'No models dir at {models_dir}')
    if (models_dir / 'final.pt').exists():
        return str(models_dir / 'final.pt')
    phase3 = sorted(glob(str(models_dir / 'phase3_*.pt')))
    if phase3:
        return phase3[-1]
    for name in ('phase2.pt', 'phase1.pt'):
        if (models_dir / name).exists():
            return str(models_dir / name)
    raise FileNotFoundError(f'No checkpoint found in {models_dir}')


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
    """
    Evaluate a saved ReW-MPC checkpoint using gradient-based planning.

    Example:
        python evaluate.py task=mw-reach seed=1 exp_name=single_task save_video=true
        python evaluate.py task=mt30 checkpoint=/path/to/model.pt eval_episodes=10
    """
    assert torch.cuda.is_available()
    assert cfg.eval_episodes > 0
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)

    if cfg.checkpoint is None:
        cfg.checkpoint = find_checkpoint(cfg)

    env = make_env(cfg)
    agent = ReWMPC(cfg)
    logger = Logger(cfg)

    assert os.path.exists(cfg.checkpoint), f'Checkpoint not found: {cfg.checkpoint}'
    agent.load(cfg.checkpoint)
    agent.model.eval()
    agent.reward_surrogate.eval()
    agent.value_surrogate.eval()
    agent.flow_prior.eval()

    print(colored(f'Task: {cfg.task_title}', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))

    scores = []
    tasks = cfg.tasks if cfg.multitask else [cfg.task]
    for task_idx, task in enumerate(tasks):
        t_idx = task_idx if cfg.multitask else None
        ep_rewards, ep_successes = [], []
        for ep_idx in range(cfg.eval_episodes):
            obs, done, ep_reward, t = env.reset(task_idx=t_idx), False, 0, 0
            frames = [] if cfg.save_video else None
            while not done:
                action = agent.act(obs, t0=(t == 0), eval_mode=True, task=t_idx)
                obs, reward, done, info = env.step(action)
                if frames is not None:
                    frames.append(env.render())
                ep_reward += reward
                t += 1
            ep_rewards.append(ep_reward)
            ep_successes.append(info.get('success', 0.))
            if frames:
                logger.save_video(frames, f'{task}_ep{ep_idx}')
        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)
        if cfg.multitask:
            scores.append(ep_successes * 100 if task.startswith('mw-') else ep_rewards / 10)
        print(colored(f'  {task:<22}\tR: {ep_rewards:.01f}  \tS: {ep_successes:.02f}', 'yellow'))

    if cfg.multitask:
        print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
    evaluate()
