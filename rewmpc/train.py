import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common.logger import Logger
from envs import make_env
from agent import ReWMPC
from trainer.offline_trainer import OfflineTrainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Offline training script for ReW-MPC.

	Example:
	    python train.py task=mt30 model_size=5 data_dir=/path/to/mt30
	    python train.py task=mt80 model_size=5 data_dir=/path/to/mt80
	"""
	assert torch.cuda.is_available(), 'CUDA required for ReW-MPC training.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer = OfflineTrainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=ReWMPC(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
