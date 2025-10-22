import copy, time
from pathlib import Path
import utils.common_utils as cu
import numpy as np
import torch
import torch.nn.functional as F

from .agent_base import BaseAgent
from .ddpg_utils import Policy, Critic, ReplayBuffer
from .ddpg_agent import DDPGAgent

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

class DDPGExtension(DDPGAgent):

    ## Your code starts here. ######
    # You need to override the update method to implement the DDPG with extensions.
    # You can modify other functions of the base class if needed.

    pass