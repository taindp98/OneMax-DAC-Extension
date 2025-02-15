import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Tuple, Optional
from onemax_dac.theory_env import OneMax
from torch import nn
from onemax_dac.dac.utils import to_tensor
import torch

def evaluate_policy(
    net: nn.Module,
    env: OneMax,
    n_eval_episodes: int,
    num_workers: int = 1,
    init_obj_rate=0.5,
    use_policy: bool = True,
    verbose: int = 1,
    verbose_tag: str = "Parallel Progress",
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
        use_policy: whether to use the policy or not
        verbose: verbosity level
    """
    problem_size = env.n
    if use_policy:
        all_states = to_tensor(
            np.array([[problem_size, fx] for fx in range(0, problem_size)]),
            device=device,
        )
        with torch.no_grad():
            q_values = net(all_states)
        action_indices = q_values.argmax(dim=1).cpu().numpy().tolist()
    else:
        action_indices = np.random.choice(
            np.arange(len(env.action_choices)),
            size=problem_size
        )
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

    outputs = Parallel(n_jobs=num_workers)(
        delayed(single_run_onell)(
            n=problem_size,
            oll_parameters=policy,
            seed=i,
            cutoff=cutoff,
            init_obj_rate=init_obj_rate,
        )
        for i in tqdm(range(n_eval_episodes), desc=verbose_tag, disable=not verbose, ncols=100)
    )
    runtimes = [output[0] for output in outputs]
    data_changes = [output[1] for output in outputs]
    costs = [output[2] for output in outputs]
    return action_indices, policy, runtimes, data_changes, costs


def get_theory_policy(problem_size: int, action_choices: list = []):
    """
    Get the theoretical policy for the given problem size
    """
    policy = []
    parameters = []
    if action_choices:
        ## discrete theory policy
        for i in range(problem_size):
            cont_lbd = np.sqrt(problem_size / (problem_size - i))
            lbd = min(action_choices, key=lambda x: abs(x - cont_lbd))
            mutation_rate = lbd / problem_size
            crossover_rate = 1 / lbd
            parameters.append([mutation_rate, lbd, crossover_rate, lbd])
            policy.append(lbd)
    else:
        ## continuous theory policy
        for i in range(problem_size):
            cont_lbd = np.sqrt(problem_size / (problem_size - i))
            floor_lbd = np.floor(cont_lbd)
            lbd = floor_lbd if cont_lbd - floor_lbd < 0.5 else floor_lbd + 1
            lbd = int(lbd)
            mutation_rate = lbd / problem_size
            crossover_rate = 1 / lbd
            parameters.append([mutation_rate, lbd, crossover_rate, lbd])
            policy.append(lbd)
    return policy, parameters


def onell_dynamic_theory(
    n,
    discrete_portfolio: list,
    seed: int,
    cutoff: int = 1e6,
    count_different_inds_only=True,
    include_xprime_crossover=True,
    init_obj_rate: float = 0.5,
):
    """
    (1+LL)-GA, dynamic version with theoretical results
    lbd = sqrt(n*(n-f(x))), p = lbd/n, c=1/lbd
    quantize the lambda values to the nearest value in the discrete_portfolio
    """
    rng = np.random.Generator(np.random.MT19937(seed))
    x = OneMax(n=n, rng=rng, init_obj_rate=init_obj_rate)
    f_x = x.fitness
    total_evals = 1
    for _ in range(int(cutoff)):
        # mutation phase
        cont_lbd = np.sqrt(n / (n - f_x))
        floor_lbd = np.floor(cont_lbd)
        lbd = floor_lbd if cont_lbd - floor_lbd < 0.5 else floor_lbd + 1
        lbd = int(lbd)
        if discrete_portfolio:
            lbd = min(discrete_portfolio, key=lambda x: abs(x - lbd))
        p = lbd / n
        ## mutation phase
        xprime, f_xprime, ne1 = x.mutate(
            p=p,
            n_childs=lbd,
            rng=rng,
        )
        ## crossover phase
        c = 1 / lbd
        y, f_y, ne2 = x.crossover(
            xprime=xprime,
            p=c,
            n_childs=lbd,
            include_xprime=include_xprime_crossover,
            count_different_inds_only=count_different_inds_only,
            rng=rng,
        )
        ## selection phase
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
    init_obj_rate,
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
    data_changes = []
    costs = []
    for _ in range(int(cutoff)):
        mutation_rate, mutation_size, crossover_rate, crossover_size = oll_parameters[f_x]
        ## mutation phase
        xprime, f_xprime, ne1 = x.mutate(p=mutation_rate, n_childs=mutation_size, rng=rng)
        ## crossover phase
        y, f_y, ne2 = x.crossover(
            xprime=xprime,
            p=crossover_rate,
            n_childs=crossover_size,
            rng=rng,
        )
        ## selection phase
        if f_x <= f_y:
            x = y
            f_x = f_y
        total_evals = total_evals + ne1 + ne2
        data_changes.append(x.data.astype(int))
        costs.append(ne1 + ne2)
        if total_evals >= cutoff or x.is_optimal():
            break
    return total_evals, data_changes, costs
