import numpy as np
from dacbench.envs.theory import BinaryProblem

class OneMax(BinaryProblem):
    """
    An individual for OneMax problem.

    The aim is to maximise the number of ones in the bitstring.
    """

    def __init__(
            self,
            n: int,
            state_dim: int = 2,
            action_choices: list = [],
            reward_choice: str = "imp_minus_evals_shifted",
            rng=np.random.default_rng(),
            init_obj_rate=None,
            **kwargs
    ):
        self.n = n
        self.init_obj_rate = init_obj_rate
        self.init_obj = int(init_obj_rate * n) if init_obj_rate is not None else None
        self.rng = rng
        self.reward_choice = reward_choice
        self.kwargs = kwargs

        if self.init_obj is None:
            super(OneMax, self).__init__(n=self.n, rng=self.rng)
        else:
            self.data = np.zeros(self.n, dtype=bool)
            random_indices = self.rng.choice(self.n, self.init_obj, replace=False)
            self.data[random_indices] = 1
            assert self.data.sum() == self.init_obj
        self.fitness = self.eval()
        self.total_evals = 1
        self.max_evals = int(0.8 * self.n * self.n)

        if action_choices:
            self.action_dim = len(action_choices)
            self.action_choices = action_choices
        else:
            self.action_dim = int(np.ceil(np.log2(n)))
            self.action_choices = [2**i for i in range(self.action_dim)]

        # print(f"🚀 Action choices: {self.action_choices}")
        self.state_dim = state_dim
        

    def eval(self):
        """
        Evaluate the individual
        """
        self.fitness = self.data.sum()
        return self.fitness

    def is_optimal(self):
        """
        Check if the individual is optimal
        """
        return self.data.all()

    def get_optimal(self):
        """
        Return the optimal solution
        """
        return self.n

    def get_fitness_after_flipping(self, locs):
        """
        Calculate the change in fitness after flipping the bits at positions locs

        Parameters
        ----------
        locs: 1d-array
            positions where bits are flipped

        Returns
        -------
            objective after flipping

        f(x_new) = f(x) + l - 2 * sum_of_flipped_block
        """
        return self.fitness + len(locs) - 2 * self.data[locs].sum()

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        return self.data[locs_x].sum() + xprime.data[locs_xprime].sum()

    def reset(self):
        if self.init_obj is None:
            super(OneMax, self).__init__(n=self.n, rng=self.rng)
        else:
            self.data = np.zeros(self.n, dtype=bool)
            random_indices = self.rng.choice(self.n, self.init_obj, replace=False)
            self.data[random_indices] = 1
            assert self.data.sum() == self.init_obj
        self.total_evals = 1
        return self.get_state(), {}

    def get_state(self):
        """Return state."""
        state = np.array([self.n, self.data.sum()])
        return state

    def step(self, action_index: int, shift: float = 0.0):
        """
        Perform the action on the individual and return the new state, reward, termination status and info
        
        """
        fitness_before_update = self.eval()

        lambda_ = self.action_choices[action_index]
        mutation_rate = np.float64(lambda_ / self.n)
        mutation_size = np.int64(lambda_)
        crossover_rate = np.float64(1.0 / lambda_)
        crossover_size = np.int64(lambda_)

        xprime, f_xprime, ne1 = self.mutate(
            p=mutation_rate,
            n_childs=mutation_size,
            rng=self.rng,
        )
        y, f_y, ne2 = self.crossover(
            xprime=xprime,
            p=crossover_rate,
            n_childs=crossover_size,
            rng=self.rng,
        )
        n_evals = ne1 + ne2
        self.total_evals += n_evals
        self.data = max([self.data, y.data], key=lambda x: sum(x))
        if self.reward_choice == "imp_div_evals":
            reward = (self.data.sum() - fitness_before_update) / n_evals
        elif self.reward_choice == "imp_minus_evals":
            reward = (self.data.sum() - fitness_before_update) - n_evals
        elif self.reward_choice == "minus_evals":
            reward = -n_evals
        elif self.reward_choice == "minus_evals_normalised":
            reward = -n_evals / self.max_evals
        elif self.reward_choice == "imp_minus_evals_normalised":
            reward = (self.data.sum() - fitness_before_update) - n_evals
            reward = reward / self.max_evals
        elif self.reward_choice == "imp":
            reward = self.data.sum() - fitness_before_update
        elif self.reward_choice == "imp_minus_evals_problem_scaled":
            reward = (self.data.sum() - fitness_before_update) - n_evals
            reward = reward / self.n
        elif self.reward_choice == "imp_minus_evals_shifted":
            reward = (self.data.sum() - fitness_before_update) - n_evals
            reward += shift
        else:
            raise ValueError("Invalid reward choice")
        truncated = False
        terminated = self.is_optimal() or (self.total_evals >= self.max_evals)
        info = {}
        return self.get_state(), reward, truncated, terminated, info