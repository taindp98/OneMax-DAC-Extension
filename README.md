<h1>
    <p align="center">
        On the Importance of Reward Design in Reinforcement Learning-based Dynamic Algorithm Configuration
    </p>
</h1>

This repository contains the implementation for paper: **On the Importance of Reward Design in Reinforcement Learning-based Dynamic Algorithm Configuration: A Case Study on OneMax with (1+($\lambda$, $\lambda$))-GA**

## 🗒️ Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#usage)
- [Results](#results)

## 💡 Introduction

We propose applying RL to control the population size of the (1+($\lambda$, $\lambda$))-GA optimizing the OneMax problem. We use the number of evaluations (#Evals) at each step to validate how well the RL-based policy can choose the proper $\lambda$ to maximize the number of 1s in a given binary string.

We provide an example to visualize improvements in a problem of size 100 (using a 10×10 grid to save space), comparing two controllers: an RL-based policy and a random policy. Blue cells denote the 1s bit, while red cells represent the 0s bit. The optimal state occurs when the grid is completely filled with blue cells.

|RL-based Policy|Random Policy|
|--|--|
|![assets/ddqn_n100.gif](assets/ddqn_n100.gif)|![assets/random_n100.gif](assets/random_n100.gif)|


## 🎯 Repository Structure

Outline the structure of repository.

```plaintext
OneMax-DAC/
├── notebooks/                     
│   └── test.ipynb                  # Testing on-the-fly using trained DDQNs
├── resources/                      # Additional resources for this project
│   ├── ddqn_ckpts                   # Contains DDQNs checkpoints for all problem sizes
│   ├── other_methods               # Contains irace-based tuning and optimal policies
├── onemax_dac/                     # Source code for the project
│   ├── train.py                    # Script to train models
│   ├── dac/                        # Main components of DAC employed in this project
│   │   ├── trainer.py              # Module to train DAC
│   │   ├── buffer.py               # Module to store the experiences
│   │   ├── agent.py                # Module to hold the environment and replay buffer
│   │   ├── policy.py               # Module of Q-Network
│   │   ├── eval.py                 # Contains functions to evaluate the policy
│   │   ├── logger.py               # Module to monitor the training process
│   │   └── utils.py                # Helping functions
|   ├── theory_env/                 # Contains theorectical environments based on DACBench
│   │   └── onemax.py               # Module of OneMax problem
├── requirements.txt                # List of dependencies
├── README.md                       # Project readme file
└── LICENSE                         # License for the project
```

## ⚙️ Installation

To re-produce this project, you will need to have the following dependencies installed:
- Ubuntu 18.04.6 LTS
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.10
- [PyTorch](https://pytorch.org/) (version 2.0 or later)

After installing Miniconda, you can create a new environment and install the required packages using the following commands:

```bash
conda create -n onemaxdac python=3.10
conda activate onemaxdac
```
For installing `torch`, refer this link: [INSTALLING PREVIOUS VERSIONS OF PYTORCH](https://pytorch.org/get-started/previous-versions/)

then clone and install dependencies:
```bash
pip install -r requirements.txt
````
## 🚀 Quickstart
### Testing
We provide the best checkpoints of DDQNs, which are trained using the best settings of reward functions in certain problem sizes at `resources/ddqn_ckpts`.

To replicate the results reported in the paper, follow the notebook [test.ipynb](notebooks/test.ipynb):
1. Initialize the DDQN and OneMax environment objects.
2. Load the trained checkpoint properly.
3. Run (1+($\lambda$, $\lambda$))-GA and observe the ERT.

**Note**: Please make sure you have the notebook kernel installed with the necessary packages.

### Training
We divide our experiments into three groups:
- Original reward function
- Reward scaling
- Reward shifting

The implementation of these families of reward functions can be found in [onemax.py](onemax_dac/theory_env/onemax.py).

### Experiment with the Original Reward Function

```bash
python onemax_dac/train.py \    ## Main Python script for training
    --problem_size 100 \        ## Set problem size n=100
    --reward_choice original \  ## Use original reward function
    --seed 1 \                  ## Set random seed
    --num_workers 4             ## Set number of CPUs for parallel processing
```

### Experiment with the Reward Scaling

```bash
python onemax_dac/train.py \    ## Main Python script for training
    --problem_size 100 \        ## Set problem size n=100
    --reward_choice scaling \  ## Use scaled reward function
    --seed 1 \                  ## Set random seed
    --num_workers 4             ## Set number of CPUs for parallel processing
```

### Experiment with the Reward Shifting

```bash
python onemax_dac/train.py \    ## Main Python script for training
    --problem_size 100 \        ## Set problem size n=100
    --reward_choice shifting \  ## Use reward shifting with fixed bias
    --fixed_shift -3 \          ## Set value of bias
    --seed 1 \                  ## Set random seed
    --num_workers 4             ## Set number of CPUs for parallel processing
```

```bash
python onemax_dac/train.py \    ## Main Python script for training
    --problem_size 100 \        ## Set problem size n=100
    --reward_choice shifting \  ## Use reward shifting with adaptive bias
    --seed 1 \                  ## Set random seed
    --num_workers 4             ## Set number of CPUs for parallel processing
```

## 📊 Results

### Terminal
After running, the terminal should look like this:

```plaintext
{'TrainingConfig': {'max_steps': 500000, 'buffer_size': 1000000, 'epsilon_start': 1.0, 'epsilon_end': 0.2, 'warmup_steps': 10000, 'batch_size': 2048, 'learning_rate': 0.001, 'gamma': 0.99, 'tau': 0.01, 'loss_fn': 'MSE', 'eval_interval': 2000, 'n_eval_episodes': 100, 'output_dir': 'outputs', 'accelerator': 'cpu', 'num_workers': 4, 'wandb': False, 'seed': 1, 'fixed_shift': None}, 'PolicyConfig': {'policy_name': 'DDQN', 'net_arch': [50, 50], 'activation_fn': 'ReLU'}, 'EnvConfig': {'problem_size': 100, 'state_dim': 2, 'discrete_action': True, 'action_choices': [], 'reward_choice': 'original', 'seed': 1, 'init_obj_rate': 0.5, 'kwargs': {}}}
Populating Buffer: 100%|████████████████████████████████████| 10000/10000 [00:03<00:00, 2512.55it/s]
[Training]: step: 14000.00 | shift: 0.00 | loss: 27.20 | best_val_rt: 689.01:   1%|▋                                                                             | 4110/490000 [00:25<50:44, 159.58it/s]
```
Observation:
- A dictionary containing the current running configurations.
- A message of `Populating Buffer` indicating that the warm-up process is running within N steps.
- Then the training process starts running with the remaining steps indicated by total steps minus warm-up steps. There is some information including: 
    - step: current evaluation step
    - shift: value of shifting
    - loss: current training loss
    - best_val_rt: best evaluated expected runtime

### Logs

During the process, we can monitor the logs by following the path `outputs/checkpoints/<date>_<time>/seed_<#>`. In this directory:

```plaintext
outputs/checkpoints/<date>_<time>/seed_<#>/
├── config.yml                      # Training configuration is stored here
├── evaluations.json                # Contains learned policies and ERTs from both evaluation and testing phases
├── best.pt                         # Best checkpoint of the Q-network
├── learning_curve.pdf              # Evaluated ERT under 100 runs during training
└── policy.pdf                      # Policy comparison
```