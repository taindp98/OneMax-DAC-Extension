class TrainingConfig:
    def __init__(
        self,
        max_steps: int = 1000000,
        buffer_size: int = 1_000_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.2,
        warmup_steps: int = 10000,
        batch_size: int = 2048,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.01,
        loss_fn: str = "MSE",
        eval_interval: int = 2000,
        n_eval_episodes: int = 100,
        output_dir: str = "outputs",
        accelerator: str = "cpu",
        num_workers: int = 1,
        wandb: bool = False,
        seed: int = 0,
        fixed_shift=None,
    ):
        self.max_steps = max_steps
        self.buffer_size = buffer_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.loss_fn = loss_fn
        self.eval_interval = eval_interval
        self.n_eval_episodes = n_eval_episodes
        self.output_dir = output_dir
        self.accelerator = accelerator
        self.num_workers = num_workers
        self.wandb = wandb
        self.seed = seed
        self.fixed_shift = fixed_shift

    def to_dict(self):
        """Convert the class attributes to a dictionary."""
        return self.__dict__
