import dataclasses
import os
import datetime
import re

import numpy as np
import pandas as pd
from termcolor import colored

from common import TASK_SET


# Remaps raw metric keys → organized wandb sections
_WANDB_REMAP = {
    'world/jepa_loss':       'loss/jepa',
    'world/sigreg_loss':     'loss/sigreg',
    'world/total_loss':      'loss/world_total',
    'world/grad_norm':       'grad/world',
    'surrogate/reward_loss': 'loss/reward_surrogate',
    'surrogate/value_loss':  'loss/value_surrogate',
    'flow/cfm_loss':         'loss/flow_cfm',
    'flow/return_mean':      'return/flow_mean',
}

# Keys that are x-axis / system info — logged as step, not as metrics
_SKIP_KEYS = {'iteration', 'elapsed_time', 'episode', 'step'}


def _wandb_key(key):
    """Map a raw metric key to its wandb section path."""
    if key in _SKIP_KEYS:
        return None
    if key in _WANDB_REMAP:
        return _WANDB_REMAP[key]
    if '+' in key:
        metric, task = key.split('+', 1)
        if metric == 'episode_success':
            return f'eval/success/{task}'
        if metric == 'episode_reward':
            return f'eval/reward/{task}'
        return None
    return f'misc/{key}'


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        print(prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v))

    observations = ", ".join([str(v) for v in cfg.obs_shape.values()])
    kvs = [
        ("task", cfg.task_title),
        ("steps", f"{int(cfg.steps):,}"),
        ("observations", observations),
        ("actions", cfg.action_dim),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


class Logger:
    """Primary logging object for ReW-MPC. Logs locally or to wandb."""

    def __init__(self, cfg):
        self._log_dir = make_dir(cfg.work_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_csv = cfg.save_csv
        self._save_agent = cfg.save_agent
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        self._eval = []
        print_run(cfg)
        self.project = cfg.get("wandb_project", "none")
        self.entity = cfg.get("wandb_entity", "none")
        if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            cfg.save_agent = False
            cfg.save_video = False
            self._wandb = None
            return
        os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
        import wandb
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=str(cfg.seed),
            group=self._group,
            tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
            dir=self._log_dir,
            config=dataclasses.asdict(cfg),
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        self._wandb = wandb

    @property
    def model_dir(self):
        return self._model_dir

    def save_agent(self, agent=None, identifier='final'):
        if self._save_agent and agent:
            fp = self._model_dir / f'{str(identifier)}.pt'
            agent.save(fp)
            if self._wandb:
                artifact = self._wandb.Artifact(
                    self._group + '-' + str(self._seed) + '-' + str(identifier),
                    type='model',
                )
                artifact.add_file(fp)
                self._wandb.log_artifact(artifact)

    def finish(self, agent=None):
        try:
            self.save_agent(agent)
        except Exception as e:
            print(colored(f"Failed to save model: {e}", "red"))
        if self._wandb:
            self._wandb.finish()

    def pprint_multitask(self, d, cfg):
        print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
        metaworld_success = []
        dmcontrol_reward = []
        for k, v in d.items():
            if '+' not in k:
                continue
            task = k.split('+')[1]
            if task in TASK_SET.get('mt30', []) and k.startswith('episode_reward'):
                dmcontrol_reward.append(v)
                print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
            elif task in TASK_SET.get('mt80', []) and task not in TASK_SET.get('mt30', []):
                if k.startswith('episode_success'):
                    metaworld_success.append(v)
                    print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))

        if metaworld_success:
            avg_s = float(np.nanmean(metaworld_success))
            d['episode_success+avg_metaworld'] = avg_s
            print(colored(f'  {"metaworld avg":<22}\tS: {avg_s:.02f}', 'yellow', attrs=['bold']))

        if dmcontrol_reward:
            avg_r = float(np.nanmean(dmcontrol_reward))
            d['episode_reward+avg_dmcontrol'] = avg_r
            print(colored(f'  {"dmcontrol avg":<22}\tR: {avg_r:.01f}', 'yellow', attrs=['bold']))

    def log(self, d, category="train"):
        assert category in ('pretrain', 'train', 'eval'), f"invalid category: {category}"

        if self._wandb:
            xkey = 'iteration' if category == 'pretrain' else 'step'
            _d = {wk: v for k, v in d.items() if (wk := _wandb_key(k)) is not None}
            if _d:
                self._wandb.log(_d, step=d[xkey])

        if category == 'eval' and self._save_csv:
            keys = ['step', 'episode_reward']
            self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
            pd.DataFrame(np.array(self._eval)).to_csv(
                self._log_dir / 'eval.csv', header=keys, index=None
            )
        # 'pretrain' console output suppressed — tqdm handles progress display
