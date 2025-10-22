"""
BaseAgent Class

This file defines the BaseAgent class, which serves as an abstract template for reinforcement learning agents. 
It provides a consistent structure and set of required methods that any agent implementation (e.g., DDPG or PPO) 
must define in order to interact properly with the environment, perform training, and manage models.

What Students Should Implement:
--------------------------------
Students are expected to fill in their own code in the SUBCLASSES of BaseAgent (e.g., DDPGAgent, PPOAgent) and implement 
the following abstract methods:

1. get_action(self, observation, evaluation=False):
    - Input: a single observation from the environment.
    - Output: an action to take, determined by the agent’s policy.
    - Behavior: should handle both training (exploration) and evaluation modes.

2. load_model(self):
    - Responsible for loading pre-trained model parameters from self.model_dir.
    - Should restore all necessary networks (policy, value, critic, etc.) for continued training or evaluation.

3. save_model(self):
    - Responsible for saving all important model parameters (policy, value, etc.) to self.model_dir.
    - Students should decide which components to store based on the algorithm used.

4. train(self):
    - The main training loop for the agent.
    - Should define how the agent collects experience, updates networks, and interacts with the environment.
    
Note that you do not need to fill in any code in the BaseAgent class. All the abstract methods MUST be implemented in
the derived classes (eg. in ddpg_agent.py etc.)
"""

class BaseAgent(object):
    def __init__(self, config=None):
        self.cfg=config["args"]
        self.env=config["env"]
        self.eval_env=config["eval_env"]
        self.action_space_dim=config["action_space_dim"] # 2
        self.observation_space_dim=config["observation_space_dim"] # 6
        self.train_device=self.cfg.device # default as "cpu"
        self.seed=config["seed"]
        self.algo_name=self.cfg.algo_name
        self.env_name=self.cfg.env_name
        self.max_action=self.cfg.max_action
        
        self.work_dir=self.cfg.work_dir # project_folder/results/env_name/algo_name/
        self.model_dir=self.cfg.model_dir
        self.logging_dir=self.cfg.logging_dir
        self.video_train_dir=self.cfg.video_train_dir
        self.video_test_dir=self.cfg.video_test_dir
        
    def get_action(self, observation, evaluation=False):
        """Given an observation, we will use this function to output an action."""
        raise NotImplementedError()
    
    def load_model(self):
        """Load the pre-trained model from the default model directory."""
        raise NotImplementedError()
    
    def save_model(self):
        """Save the trained models to the default model directory, for example, your value network
        and policy network. However, it depends on your agent/algorithm to decide what kinds of models
        to store."""
        raise NotImplementedError()
    
    def train(self):
        """Train the RL agent"""
        raise NotImplementedError()