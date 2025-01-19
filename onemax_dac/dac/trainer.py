import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from onemax_dac.dac import (
    QNetwork,
    Logger,
    Agent,
    evaluate_policy,
    single_run_onell,
    onell_dynamic_theory,
)
from joblib import Parallel, delayed
import pandas as pd
from onemax_dac.dac.utils import to_tensor, soft_update
from onemax_dac.configs import TrainingConfig
import json


class OneMaxDAC:
    def __init__(
        self,
        agent: Agent,
        q_online: QNetwork,
        q_target: QNetwork,
        logger: Logger,
        rng: np.random.default_rng,
        ckpt_dir: str,
        training_config: TrainingConfig,
    ):
        """
        OneMaxDAC Trainer
        Args:
            agent: Agent object
            q_online: QNetwork object
            q_target: QNetwork object
            logger: Logger object
            rng: random number generator
            ckpt_dir: checkpoint directory
            training_config: training configuration
        """
        self.training_config = training_config
        if self.training_config.accelerator == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.q_online = q_online.to(self.device)
        self.q_target = q_target.to(self.device)
        self.agent = agent
        self.logger = logger
        self.rng = rng
        self.training_config = training_config

        self.optimizer = optim.Adam(
            self.q_online.parameters(),
            lr=self.training_config.learning_rate,
        )
        self.ckpt_dir = ckpt_dir
        if self.training_config.loss_fn == "MSE":
            self.loss_fnc = F.mse_loss
        else:
            self.loss_fnc = F.smooth_l1_loss

        self.populate()

    def populate(self):
        """
        Populate the replay buffer with random actions
        """
        for _ in tqdm(
            range(self.training_config.warmup_steps), desc="Populating Buffer", ncols=100
        ):
            self.agent.play_step(
                net=self.q_online,
                shift=0,
                epsilon=self.training_config.epsilon_start,
                device=self.device,
            )

        if "shifted" in self.agent.env.reward_choice:
            if self.training_config.fixed_shift:
                self.shift = float(self.training_config.fixed_shift)
            else:
                self.shift = -self.agent.replay_buffer.get_reward_stats(mode="median") / 5
        else:
            self.shift = 0

        self.logger.log_json(
            phase="settings",
            shift=self.shift,
            action_choices=self.agent.env.action_choices,
        )

    def training_step(self, batch_size: int = 2048):
        """
        Perform a single training step
        Args:
            batch_size: batch size for training
        """
        data_batch = self.agent.replay_buffer.random_next_batch(
            batch_size,
        )
        (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_terminal_flags,
        ) = (
            to_tensor(data_batch[0], device=self.device),
            to_tensor(data_batch[1], device=self.device),
            to_tensor(data_batch[2], device=self.device),
            to_tensor(data_batch[3], device=self.device),
            to_tensor(data_batch[4], device=self.device),
        )

        target = (
            batch_rewards
            + (1 - batch_terminal_flags)
            * self.training_config.gamma
            * self.q_target(batch_next_states)[
                torch.arange(batch_size).long(),
                torch.argmax(self.q_online(batch_next_states), dim=1),
            ]
        )

        current_prediction = self.q_online(batch_states)[
            torch.arange(batch_size).long(), batch_actions.long()
        ]

        loss = self.loss_fnc(current_prediction, target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        soft_update(self.q_target, self.q_online, self.training_config.tau)
        return loss

    def learn(self, verbose: int = 1):
        training_steps = self.training_config.max_steps - self.training_config.warmup_steps
        progress_bar = tqdm(range(training_steps), ncols=200, disable=not verbose)

        best_val_rt = np.inf
        for step in progress_bar:
            self.agent.play_step(
                net=self.q_online,
                shift=self.shift,
                epsilon=self.training_config.epsilon_end,
                device=self.device,
            )
            loss = self.training_step(
                batch_size=self.training_config.batch_size,
            )
            if step % self.training_config.eval_interval == 0:
                action_indices, policy, runtimes = evaluate_policy(
                    net=self.q_online,
                    env=self.agent.env,
                    n_eval_episodes=self.training_config.n_eval_episodes,
                    num_workers=self.training_config.num_workers,
                    init_obj_rate=self.agent.env.init_obj_rate,
                    verbose=False,
                    device=self.device,
                )
                ## csv logging action_indices, policy, runtimes, episode, step
                self.logger.log_json(
                    phase="trainval",
                    episode=self.agent.total_episodes,
                    step=step + self.training_config.warmup_steps,
                    action_indices=action_indices,
                    policy=policy,
                    runtimes=runtimes,
                )
                mean_val_rt = np.mean(runtimes)
                if mean_val_rt < best_val_rt:
                    best_val_rt = mean_val_rt
                    torch.save(self.q_online.state_dict(), os.path.join(self.ckpt_dir, "best.pt"))
                self.logger.update(
                    step=step + self.training_config.warmup_steps,
                    shift=self.shift,
                    loss=loss,
                    best_val_rt=best_val_rt,
                )
                progress_bar.set_description(f"[Training]: {self.logger}")

        self.test(k=5, n_test_episodes=1000, verbose=verbose)
        self.logger.close()

    def __repr__(self):
        return "onemaxdac"

    def test(self, k: int = 5, n_test_episodes: int = 1000, verbose: int = -1):
        ## load evaluation data and convert to pandas dataframe
        json_logdata = json.load(open(os.path.join(self.ckpt_dir, "evaluations.json")))
        trainval_data = pd.DataFrame(json_logdata["trainval"])
        ## get top-k policy by min of average runtimes column
        runtimes = trainval_data["runtimes"].tolist()
        mean_runtimes = [np.mean(runtime) for runtime in runtimes]
        top_k_min_indices = np.argsort(mean_runtimes)[:k]
        ## get top-k policies
        top_k_policies = trainval_data.loc[top_k_min_indices, "policy"].tolist()
        top_k_steps = trainval_data.loc[top_k_min_indices, "step"].tolist()
        init_obj_rate = self.agent.env.init_obj_rate
        problem_size = self.agent.env.n
        cutoff = int(0.8 * problem_size * problem_size)

        ## compute continuous and discrete policy runtimes of theory
        continuous_runtimes = Parallel(n_jobs=self.training_config.num_workers)(
            delayed(onell_dynamic_theory)(
                n=self.agent.env.n,
                discrete_portfolio=[],
                seed=i,
                cutoff=cutoff,
                init_obj_rate=init_obj_rate,
            )
            for i in tqdm(
                range(n_test_episodes), desc="Parallel Progress", disable=not verbose, ncols=100
            )
        )
        self.logger.log_json(
            phase="test",
            method="continuous_theory",
            step=0,
            mean_runtime=np.mean(continuous_runtimes),
            std_runtime=np.std(continuous_runtimes),
            runtimes=continuous_runtimes,
        )
        discrete_runtimes = Parallel(n_jobs=self.training_config.num_workers)(
            delayed(onell_dynamic_theory)(
                n=self.agent.env.n,
                discrete_portfolio=self.agent.env.action_choices,
                seed=i,
                cutoff=cutoff,
                init_obj_rate=init_obj_rate,
            )
            for i in tqdm(
                range(n_test_episodes), desc="Parallel Progress", disable=not verbose, ncols=100
            )
        )
        self.logger.log_json(
            phase="test",
            method="discrete_theory",
            step=0,
            mean_runtime=np.mean(discrete_runtimes),
            std_runtime=np.std(discrete_runtimes),
            runtimes=discrete_runtimes,
        )
        ## compute runtimes of top-k policies
        for idx, policy in enumerate(top_k_policies):
            runtimes = Parallel(n_jobs=self.training_config.num_workers)(
                delayed(single_run_onell)(
                    n=self.agent.env.n,
                    oll_parameters=policy,
                    seed=i,
                    cutoff=cutoff,
                    init_obj_rate=init_obj_rate,
                )
                for i in tqdm(
                    range(n_test_episodes),
                    desc="Parallel Progress",
                    disable=not verbose,
                    ncols=100,
                )
            )
            self.logger.log_json(
                phase="test",
                method="dac",
                step=top_k_steps[idx],
                mean_runtime=np.mean(runtimes),
                std_runtime=np.std(runtimes),
                runtimes=runtimes,
            )
