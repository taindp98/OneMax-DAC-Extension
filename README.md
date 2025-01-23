<h1>
    <p align="center">
        On the Importance of Reward Design in Reinforcement Learning-based Dynamic Algorithm Configuration
    </p>
</h1>

This repository contains PyTorch implementation for our paper: **On the Importance of Reward Design in Reinforcement Learning-based Dynamic Algorithm Configuration: A Case Study on OneMax with (1+($\lambda$,$\lambda$))-GA**

## Table of Contents

- [Introduction](#introduction)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#usage)
- [Examples](#examples)
- [Algorithms](#algorithms)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

We propose applying RL to control the population size of the (1+($\lambda$,$\lambda$))-GA optimizing OneMax problem.

## Repository Structure

Outline the structure of repository.

```plaintext
OneMax-DAC/
├── notebooks/                      # Running examples
│   ├── train.ipynb
│   └── test.ipynb
├── onemax_dac/                     # Source code for the project
│   ├── __init__.py
│   ├── train.py                    # Script to train models
│   ├── evaluate.py                 # Script to evaluate models
│   ├── dac/                        # Main components of DAC employed in this project
│   │   ├── trainer.py              # Module to train DAC
│   │   ├── buffer.py               # Module to store the experiences
│   │   ├── agent.py                # Module to hold the environment and replay buffer
│   │   ├── policy.py               # Module of Q-Network
│   │   ├── eval.py                 # Contains functions to evaluate the policy
│   │   ├── logger.py               # Module to monitor the training process
│   │   └── utils.py                # Helping functions
|   ├── env/                        # Contains theorectical environments based on DACBench
│   │   └── onemax.py               # Module of OneMax problem
├── requirements.txt                # List of dependencies
├── README.md                       # Project readme file
└── LICENSE                         # License for the project
````

## Installation

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
## Quickstart
