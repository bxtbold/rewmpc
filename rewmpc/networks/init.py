import torch.nn as nn


def weight_init(m):
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)
	elif isinstance(m, nn.GRU):
		for name, param in m.named_parameters():
			if 'weight' in name:
				nn.init.orthogonal_(param)
			elif 'bias' in name:
				nn.init.constant_(param, 0)


def zero_(params):
	for p in params:
		p.data.fill_(0)
