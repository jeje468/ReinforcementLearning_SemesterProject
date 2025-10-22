""" This file define some components for DDPG algorithm, including:
    - Actor network (Policy)
    - Critic network
    - Replay buffer

The student should complete the code in the middle for these components.

Hint: refer to the course exercises."""

from collections import namedtuple, defaultdict
import pickle, os, random
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, Independent

Batch = namedtuple('Batch', ['state', 'action', 'next_state', 'reward', 'not_done', 'extra'])

# Actor-critic agent
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        

    def forward(self, state):
        return 


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        True

    def forward(self, state, action):
        return

class ReplayBuffer(object):
    def __init__(self, state_shape:tuple, action_dim: int, max_size=int(1e6)):
        True
