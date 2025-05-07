# DQN-LunarLander 
### This project implements a Deep Q-Network (DQN) to solve the LunarLander-v3 environment using PyTorch and Gymnasium. As an extension, we encorporate Double DQN to reduce overestimation and improve learning stability.

# Environment Setup
## Requirements:
###            - Python 3.10
###            - PyTorch
###            - Gymnasium
###            - Matplotlib
###            - NumPy

## Create virtual environment
`python3 -m venv dqn_env`
####
`source dqn_env/bin/activate`
## Install Dependencies
`pip install -upgrade pip`
`pip install torch gymnasium[box2d] matplotlib numpy`
## Run training
`python main.py`
