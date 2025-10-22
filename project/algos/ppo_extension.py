"""
PPOExtension - Dual-Clip PPO Assignment (Graduate Research Task)
----------------------------------------------------------------
This file defines an extended PPO agent template for implementing
the "Dual-Clip PPO" algorithm (refer to: Ye et al., 2020).

Your tasks:
1. Implement the return computation (GAE or simple discounted returns).
2. Implement the minibatch loop in `ppo_epoch()`.
3. Implement the modified PPO update with a dual clipping mechanism.
4. Think critically about how dual clipping modifies the policy loss.

All key sections are marked with:
    # ===== YOUR CODE STARTS HERE =====
    # ===== YOUR CODE ENDS HERE =====
"""

from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time


class PPOExtension(PPOAgent):
    def __init__(self, config=None):
        super(PPOAgent, self).__init__(config)
        self.device = self.cfg.device
        self.policy = Policy(self.observation_space_dim, self.action_space_dim, self.env).to(self.device)
        self.lr = self.cfg.lr
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(self.lr))
        self.batch_size = self.cfg.batch_size
        self.gamma = self.cfg.gamma
        self.tau = self.cfg.tau
        self.clip = self.cfg.clip
        self.epochs = self.cfg.epochs
        self.running_mean = None
        self.states, self.actions, self.next_states = [], [], []
        self.rewards, self.dones, self.action_log_probs = [], [], []
        self.silent = self.cfg.silent

    def update_policy(self):
        """Perform multiple PPO updates over collected rollouts."""
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.next_states = torch.stack(self.next_states)
        self.rewards = torch.stack(self.rewards).squeeze()
        self.dones = torch.stack(self.dones).squeeze()
        self.action_log_probs = torch.stack(self.action_log_probs).squeeze()

        for e in range(self.epochs):
            self.ppo_epoch()

        # Clear rollout buffers
        self.states, self.actions, self.next_states = [], [], []
        self.rewards, self.dones, self.action_log_probs = [], [], []

    def compute_returns(self):
        """
        Compute the discounted returns and advantages (GAE) for Dual-Clip PPO.

        Expected:
        - Incorporate γ (discount factor) and τ (GAE parameter)
        - Bootstrap with critic values
        - Return the target values for the critic
        """
        # ===== YOUR CODE STARTS HERE =====
        # Steps:
        # 1. Evaluate value and next-value predictions from self.policy.
        # 2. Compute δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        # 3. Compute GAE recursively backwards in time.
        # 4. Return torch.Tensor of reversed returns.
        raise NotImplementedError("Implement return computation for Dual-Clip PPO.")
        # ===== YOUR CODE ENDS HERE =====

    def ppo_epoch(self):
        """
        Run one full PPO epoch (mini-batch sampling and updates).
        """
        # ===== YOUR CODE STARTS HERE =====
        # Steps:
        # 1. Generate all indices and compute returns via self.compute_returns().
        # 2. Randomly sample batches of size self.batch_size.
        # 3. For each batch, call self.ppo_update().
        # 4. Remove used indices until none remain.
        raise NotImplementedError("Implement epoch-wise minibatch update logic.")
        # ===== YOUR CODE ENDS HERE =====

    def ppo_update(self, states, actions, rewards, next_states, dones, old_log_probs, targets):
        """
        Implement the Dual-Clip PPO loss function and optimization step.

        Key formulas:
        - ratio = exp(new_log_prob − old_log_prob)
        - clipped surrogate loss:
              L_clip = min(ratio * A, clip(ratio, 1−ε, 1+ε) * A)
        - Dual clipping introduces an additional term:
              L_dual = max(L_clip, c * A)   for negative advantages (A < 0)
          where c > 1 is the dual-clip threshold (hyperparameter).
        """
        # ===== YOUR CODE STARTS HERE =====
        # 1. Forward pass: compute new log probabilities and value estimates.
        # 2. Compute the probability ratio.
        # 3. Compute normalized advantages (A = target − value, normalized).
        # 4. Implement standard PPO clipped loss.
        # 5. Extend to Dual-Clip PPO by applying the dual clipping rule for A < 0.
        # 6. Add value loss and entropy regularization.
        # 7. Combine into total loss, backpropagate, and update parameters.
        raise NotImplementedError("Implement Dual-Clip PPO loss and optimization.")
        # ===== YOUR CODE ENDS HERE =====

    def get_action(self, observation, evaluation=False):
        """Select an action from the current policy."""
        # ===== YOUR CODE STARTS HERE =====
        # Convert observation to tensor, pass through policy,
        # sample or take mean action depending on evaluation flag.
        # Return both the action and its log probability.
        raise NotImplementedError("Implement action sampling for Dual-Clip PPO.")
        # ===== YOUR CODE ENDS HERE =====

    def store_outcome(self, state, action, next_state, reward, action_log_prob, done):
        """Store one transition into the buffer."""
        # ===== YOUR CODE STARTS HERE =====
        # Append each element (as torch.Tensor) to self.states, self.actions, etc.
        raise NotImplementedError("Implement transition storage.")
        # ===== YOUR CODE ENDS HERE =====

    def train_iteration(self, ratio_of_episodes):
        """Run one environment episode and update policy when enough samples are collected."""
        # ===== YOUR CODE STARTS HERE =====
        # Steps:
        # 1. Reset the environment.
        # 2. Collect transitions until done or max steps.
        # 3. Call self.update_policy() periodically.
        # 4. Adjust policy exploration using self.policy.set_logstd_ratio().
        raise NotImplementedError("Implement one training iteration for Dual-Clip PPO.")
        # ===== YOUR CODE ENDS HERE =====

    def train(self):
        """Overall training loop for multiple episodes."""
        # ===== YOUR CODE STARTS HERE =====
        # 1. Initialize logger if needed.
        # 2. Loop over training episodes, calling train_iteration().
        # 3. Track average returns, log results, and save models periodically.
        raise NotImplementedError("Implement the high-level training loop.")
        # ===== YOUR CODE ENDS HERE =====

    def load_model(self):
        """Load model weights."""
        # ===== YOUR CODE STARTS HERE =====
        raise NotImplementedError("Implement model loading.")
        # ===== YOUR CODE ENDS HERE =====

    def save_model(self):
        """Save model weights."""
        # ===== YOUR CODE STARTS HERE =====
        raise NotImplementedError("Implement model saving.")
        # ===== YOUR CODE ENDS HERE =====