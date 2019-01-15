class Config:
    def __init__(self):
        self.device = 'cpu'
        self.seed = 1
        self.random_seed = 1
        self.num_agents = 1

        self.tau = 1e-3
		# critic weight decay
        self.weight_decay = 0.
        self.states = None
        self.state_size = None
        self.action_size = None
        #self.learning_rate = 0.001
        self.lr_actor = 1e-4
        self.lr_critic = 3e-4
        self.batch_size = 512
		# memory buffer size
        self.buffer_size = int(1e6)
        self.gamma = 0.999
        self.update_every = 4
        self.gradient_clip = None
        self.entropy_weight = 0.01
        self.eps_start = 1.0
        self.eps_end = 0.001
        self.eps_decay = 0.995
