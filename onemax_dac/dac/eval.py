import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Tuple, Optional
from onemax_dac.env import OneMax
from torch import nn
from onemax_dac.dac.utils import to_tensor
import torch

def evaluate_policy(
    net: nn.Module,
    env: OneMax,
    n_eval_episodes: int,
    num_workers: int = 1,
    init_obj_rate = 0.5,
    verbose: int = 1,
    device: torch.device = torch.device("cpu"),
):
    """
    Evaluate the policy for a given number of episodes
    Args:
        net: QNetwork
        env: OneMax environment
        n_eval_episodes: number of episodes to evaluate
        num_workers: number of parallel workers
        init_obj_rate: initial objective rate
        verbose: verbosity level
    """
    problem_size = env.n
    all_states = to_tensor(
        np.array([[problem_size, fx] for fx in range(0, problem_size)]),
        device=device,
    )
    q_values = net(all_states)
    action_indices = q_values.argmax(dim=1).cpu().numpy().tolist()
    policy = []
    for fitness, lambda_index in enumerate(action_indices):
        lambda_ = env.action_choices[lambda_index]
        mutation_rate = lambda_ / problem_size
        crossover_rate = 1.0 / lambda_
        policy.append(
            [
                np.float64(mutation_rate),
                np.int64(lambda_),
                np.float64(crossover_rate),
                np.int64(lambda_),
            ]
        )
    cutoff = int(0.8 * problem_size * problem_size)

    runtimes = Parallel(n_jobs=num_workers)(
        delayed(single_run_onell)(
            n=problem_size,
            oll_parameters=policy,
            seed=i,
            cutoff=cutoff,
            init_obj_rate=init_obj_rate
        )
        for i in tqdm(
            range(n_eval_episodes),
            desc="Parallel Progress",
            disable=not verbose,
            ncols=100
        )
    )
    return action_indices, policy, runtimes

def onell_dynamic_theory(
    n,
    discrete_portfolio: list,
    seed: int,
    cutoff: int = 1e6,
    count_different_inds_only=True,
    include_xprime_crossover=True,
    init_obj_rate: float = 0.0,
):
    """
    (1+LL)-GA, dynamic version with theoretical results
    lbd = sqrt(n*(n-f(x))), p = lbd/n, c=1/lbd
    quantize the lambda values to the nearest value in the discrete_portfolio
    """
    rng = np.random.Generator(np.random.MT19937(seed))
    init_obj = int(init_obj_rate * n) if init_obj_rate is not None else None
    x = OneMax(n=n, rng=rng, init_obj=init_obj)
    f_x = x.fitness
    # total number of solution evaluations
    total_evals = 1
    for _ in range(int(cutoff)):
        # mutation phase
        lbd = np.sqrt(n / (n - f_x))
        ## quantize the lambda values to the nearest value in the discrete_portfolio
        if discrete_portfolio:
            lbd = min(discrete_portfolio, key=lambda x: abs(x - lbd))
        p = lbd / n
        xprime, f_xprime, ne1 = x.mutate(
            p,
            round(lbd),
            rng=rng,
        )
        # crossover phase
        c = 1 / lbd
        y, f_y, ne2 = x.crossover(
            xprime,
            c,
            round(lbd),
            include_xprime_crossover,
            count_different_inds_only,
            rng=rng,
        )
        # selection phase
        if f_x <= f_y:
            x = y
            f_x = f_y

        total_evals = total_evals + ne1 + ne2
        if total_evals >= cutoff or x.is_optimal():
            break
    return total_evals


def single_run_onell(
    n: int,
    oll_parameters: List[Tuple[float, int, float, int]],
    seed: int,
    cutoff: int,
    init_obj_rate
):
    """
    Single run of (1+LL)-GA
    Args:
        n: problem size
        oll_parameters: list of tuples of mutation_rate, mutation_size, crossover_rate, crossover_size
        seed: random seed
        cutoff: maximum number of evaluations
        init_obj_rate: initial objective rate
    """
    rng = np.random.Generator(np.random.MT19937(seed))
    x = OneMax(n=n, rng=rng, init_obj_rate=init_obj_rate)
    f_x = x.fitness
    # total number of solution evaluations
    total_evals = 1
    for _ in range(int(cutoff)):
        mutation_rate, mutation_size, crossover_rate, crossover_size = oll_parameters[
            f_x
        ]
        # mutation phase
        xprime, f_xprime, ne1 = x.mutate(
            p=mutation_rate, n_childs=mutation_size, rng=rng
        )
        # crossover phase
        y, f_y, ne2 = x.crossover(
            xprime=xprime,
            p=crossover_rate,
            n_childs=crossover_size,
            rng=rng,
        )
        # selection phase
        if f_x <= f_y:
            x = y
            f_x = f_y
        total_evals = total_evals + ne1 + ne2

        if total_evals >= cutoff or x.is_optimal():
            break
    return total_evals