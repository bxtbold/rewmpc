class Trainer:
	"""Base trainer class for ReW-MPC."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger
		print('Architecture:\n', self.agent.model)

	def eval(self):
		raise NotImplementedError

	def train(self):
		raise NotImplementedError
