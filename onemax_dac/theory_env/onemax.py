"""Theory Environment."""

from copy import deepcopy
import numpy as np


class BinaryProblem:
    """An abstract class for an individual in binary representation."""

    def __init__(self, n, rng=None):
        """Init problem."""
        if rng is None:
            rng = np.random.default_rng()
        self.data = rng.choice([True, False], size=n)
        self.n = n
        self.fitness = self.eval()

    def initialise_with_fixed_number_of_bits(self, k, rng=None):
        """Init with given number of bits."""
        if rng is None:
            rng = np.random.default_rng()
        nbits = self.data.sum()
        if nbits < k:
            ids = rng.choice(np.where(self.data is False)[0], size=k - nbits, replace=False)
            self.data[ids] = True
            self.eval()

    def is_optimal(self):
        """Get is_optimal flag."""

    def get_optimal(self):
        """Get optimum."""

    def eval(self):
        """Evaluate fitness."""

    def get_fitness_after_flipping(self, locs):
        """Calculate the change in fitness after flipping the bits at positions locs.

        Parameters
        ----------
        locs: 1d-array
            positions where bits are flipped

        Returns:
        -------
            objective after flipping

        """
        raise NotImplementedError

    def get_fitness_after_crossover(self, xprime, locs_x, locs_xprime):
        """Calculate fitness of the child aftering being crossovered with xprime.

        Parameters
        ----------
        xprime: 1d boolean array
            the individual to crossover with
        locs_x: 1d boolean/integer array
            positions where we keep current bits of self
        locs_xprime: : 1d boolean/integer array
            positions where we change to xprime's bits

        Returns:
        -------
            fitness of the new individual after crossover

        """
        raise NotImplementedError

    def flip(self, locs):
        """Flip the bits at position indicated by locs.

        Parameters
        ----------
        locs: 1d-array
            positions where bits are flipped

        Returns:
        -------
            the new individual after the flip

        """
        child = deepcopy(self)
        child.data[locs] = ~child.data[locs]
        child.eval()
        return child

    def combine(self, xprime, locs_xprime):
        """Combine (crossover) self and xprime by taking xprime's bits at locs_xprime
        and self's bits at other positions.

        Parameters
        ----------
        xprime: 1d boolean array
            the individual to crossover with
        locs_x: 1d boolean/integer array
            positions where we keep current bits of self
        locs_xprime: : 1d boolean/integer array
            positions where we change to xprime's bits

        Returns:
        -------
            the new individual after the crossover

        """
        child = deepcopy(self)
        child.data[locs_xprime] = xprime.data[locs_xprime]
        child.eval()
        return child

    def mutate(self, p, n_childs, rng=None):
        """Draw l ~ binomial(n, p), l>0.

        Generate n_childs children by flipping exactly l bits

        Returns:
        -------
            the best child (maximum fitness), its fitness and number of evaluations used

        """
        if rng is None:
            rng = np.random.default_rng()
        assert p >= 0

        if p == 0:
            return self, self.fitness, 0

        length = 0
        while length == 0:
            length = rng.binomial(self.n, p)

        best_obj = -1
        best_locs = None
        for _i in range(n_childs):
            locs = rng.choice(self.n, size=length, replace=False)
            obj = self.get_fitness_after_flipping(locs)
            if obj > best_obj:
                best_locs = locs
                best_obj = obj

        best_child = self.flip(best_locs)

        return best_child, best_child.fitness, n_childs

    def mutate_rls(self, length, rng=None):
        """Generate a child by flipping exactly l bits.

        Returns:
        -------
            child, its fitness

        """
        if rng is None:
            rng = np.random.default_rng()
        assert length >= 0

        if length == 0:
            return self, self.fitness, 0

        locs = rng.choice(self.n, size=length, replace=False)
        child = self.flip(locs)

        return child, child.fitness, 1

    def crossover(
        self,
        xprime,
        p,
        n_childs,
        include_xprime=True,
        count_different_inds_only=True,
        rng=None,
    ):
        """Crossover operation in population.

        Crossover operator: for each bit, taking value from x with probability p
        and from self with probability 1-p

        Parameters
        ----------
        xprime
            the individual to crossover with
        p : float
            probability in [0,1]
        n_childs : int
            number of child individuals
        include_xprime : bool
            whether to inculde x
        count_different_inds_only : bool
            whether to only count different individuals
        rng:
            random number generator

        """
        if rng is None:
            rng = np.random.default_rng()
        assert p <= 1

        if p == 0:
            if include_xprime:
                return xprime, xprime.fitness, 0
            return self, self.fitness, 0

        best_obj = xprime.fitness if include_xprime else -1
        best_locs = None

        n_evals = 0
        ls = rng.binomial(self.n, p, size=n_childs)
        for l in ls:
            locs_xprime = rng.choice(self.n, l, replace=False)
            locs_x = np.full(self.n, True)
            locs_x[locs_xprime] = False
            obj = self.get_fitness_after_crossover(xprime, locs_x, locs_xprime)

            if (
                obj not in (self.fitness, xprime.fitness)
                or (not np.array_equal(xprime.data[locs_xprime], self.data[locs_xprime]))
                and (not np.array_equal(self.data[locs_x], xprime.data[locs_x]))
            ):
                n_evals += 1

            if obj > best_obj:
                best_obj = obj
                best_locs = locs_xprime

        child = self.combine(xprime, best_locs) if best_locs is not None else xprime

        if not count_different_inds_only:
            n_evals = n_childs

        return child, child.fitness, n_evals


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
        reward_choice: str = "original",
        rng=np.random.default_rng(),
        init_obj_rate=None,
        **kwargs,
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

        if self.reward_choice == "original":
            reward = (self.data.sum() - fitness_before_update) - n_evals
        elif self.reward_choice == "scaling":
            reward = (self.data.sum() - fitness_before_update) - n_evals
            reward = reward / self.n
        elif self.reward_choice == "shifting":
            reward = (self.data.sum() - fitness_before_update) - n_evals
            reward += shift
        elif self.reward_choice == "scaling_shifting":
            reward = (self.data.sum() - fitness_before_update) - n_evals
            reward = reward / self.n
            reward += shift
        else:
            raise ValueError("Invalid reward choice")
        truncated = False
        terminated = self.is_optimal() or (self.total_evals >= self.max_evals)
        info = {}
        return self.get_state(), reward, truncated, terminated, info
