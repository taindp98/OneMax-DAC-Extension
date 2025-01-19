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
- [Usage](#usage)
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
|   ├── envs/                       # Contains theorectical environments based on DACBench
│   │   └── onemax.py               # Module of OneMax problem
├── requirements.txt                # List of dependencies
├── README.md                       # Project readme file
└── LICENSE                         # License for the project
````

## Installation

Step-by-step instructions to install the necessary dependencies and set up the project. Include any prerequisites and how to install them.

```bash
# Clone the repository
git clone https://github.com/taindp98/OneMax-DAC.git

# Navigate to the project directory
cd OneMax-DAC

# Install required packages
pip install -r requirements.txt