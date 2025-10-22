"""
PPOAgent - Graduate-Level Assignment
------------------------------------
This file defines a PPO (Proximal Policy Optimization) agent framework.

Your tasks:
1. Implement the GAE-based return computation in `compute_returns()`.
2. Implement the mini-batch iteration in `ppo_epoch()`.
3. Implement the PPO clipped loss in `ppo_update()`.

Each function includes clear placeholders:
    # ===== YOUR CODE STARTS HERE =====
    # ===== YOUR CODE ENDS HERE =====

Reference: Schulman et al., “Proximal Policy Optimization Algorithms,” 2017.
"""

from .agent_base import BaseAgent
from .ppo_utils import Policy
import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time


class PPOAgent(BaseAgent):
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
        """Perform multiple epochs of PPO updates over collected rollouts."""
        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)
        self.next_states = torch.stack(self.next_states)
        self.rewards = torch.stack(self.rewards).squeeze()
        self.dones = torch.stack(self.dones).squeeze()
        self.action_log_probs = torch.stack(self.action_log_probs).squeeze()

        for e in range(self.epochs):
            self.ppo_epoch()

        # Clear replay buffers after policy update
        self.states, self.actions, self.next_states = [], [], []
        self.rewards, self.dones, self.action_log_probs = [], [], []

    def compute_returns(self):
        """
        Compute the returns (targets) and optionally Generalized Advantage Estimation (GAE).

        Expected:
        - Use self.gamma (discount factor)
        - Use self.tau (GAE parameter)
        - Use values and next_values predicted by the critic
        - Return a torch.Tensor of shape [T] containing discounted returns

        Hints:
        - returns_t = reward_t + gamma * (1 - done_t) * next_value_t
        - For GAE: gae_t = delta_t + gamma * tau * (1 - done_t) * gae_{t+1}
        - delta_t = reward_t + gamma * next_value_t * (1 - done_t) - value_t
        """
        # ===== YOUR CODE STARTS HERE =====
        # You can use the self.policy to get values for self.states and self.next_states.
        # Initialize gae = 0, and iterate backwards over timesteps.
        # Store "gae + value_t" in a list, then reverse it at the end.
        raise NotImplementedError("Implement the return computation using GAE.")
        # ===== YOUR CODE ENDS HERE =====

    def ppo_epoch(self):
        """
        Run one full PPO epoch (multiple minibatch updates).
        Sample random minibatches from collected trajectories and call ppo_update().
        """
        # ===== YOUR CODE STARTS HERE =====
        # Steps:
        # 1. Compute the returns using self.compute_returns()
        # 2. Shuffle indices
        # 3. While enough samples remain, draw a minibatch (size self.batch_size)
        # 4. Call self.ppo_update() with the selected minibatch
        # 5. Remove those indices from the pool
        raise NotImplementedError("Implement minibatch sampling and PPO update iteration.")
        # ===== YOUR CODE ENDS HERE =====

    def ppo_update(self, states, actions, rewards, next_states, dones, old_log_probs, targets):
        """
        Implement the core PPO update:
        - Compute policy loss (clipped surrogate objective)
        - Compute value loss
        - Add entropy regularization
        - Combine into total loss and optimize

        Key variables:
        - ratio = exp(new_log_prob - old_log_prob)
        - clipped_ratio = clamp(ratio, 1 - clip, 1 + clip)
        - advantage = target - value
        """
        # ===== YOUR CODE STARTS HERE =====
        # 1. Forward pass: get new action distribution and value predictions
        # 2. Compute log probs for actions and the ratio
        # 3. Normalize advantages
        # 4. Compute policy objective: min(ratio * adv, clipped_ratio * adv)
        # 5. Compute value loss and entropy
        # 6. Combine total loss and perform optimizer step
        raise NotImplementedError("Implement the PPO clipped objective and optimization step.")
        # ===== YOUR CODE ENDS HERE =====

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.device)
        action_dist, _ = self.policy.forward(x)
        if evaluation:
            action = action_dist.mean.detach()
        else:
            action = action_dist.sample()
        aprob = action_dist.log_prob(action)
        return action, aprob

    def store_outcome(self, state, action, next_state, reward, action_log_prob, done):
        self.states.append(torch.from_numpy(state).float())
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob.detach())
        self.rewards.append(torch.Tensor([reward]).float())
        self.dones.append(torch.Tensor([done]))
        self.next_states.append(torch.from_numpy(next_state).float())

    def train_iteration(self, ratio_of_episodes):
        """Run one episode of interaction and optionally update the policy."""
        reward_sum, episode_length, num_updates = 0, 0, 0
        done = False
        observation, _ = self.env.reset()

        while not done and episode_length < self.cfg.max_episode_steps:
            action, action_log_prob = self.get_action(observation)
            prev_obs = observation.copy()
            observation, reward, done, _, _ = self.env.step(action)

            self.store_outcome(prev_obs, action, observation, reward, action_log_prob, done)
            reward_sum += reward
            episode_length += 1

            if len(self.states) > self.cfg.min_update_samples:
                self.update_policy()
                num_updates += 1
                self.policy.set_logstd_ratio(ratio_of_episodes)

        return {'episode_length': episode_length, 'ep_reward': reward_sum}

    def train(self):
        """Top-level training loop."""
        if self.cfg.save_logging:
            L = cu.Logger()
        total_step, run_episode_reward = 0, []
        start = time.perf_counter()

        for ep in range(self.cfg.train_episodes + 1):
            ratio_of_episodes = (self.cfg.train_episodes - ep) / self.cfg.train_episodes
            train_info = self.train_iteration(ratio_of_episodes)
            train_info.update({'episodes': ep})
            total_step += train_info['episode_length']
            train_info.update({'total_step': total_step})
            run_episode_reward.append(train_info['ep_reward'])
            logstd = self.policy.actor_logstd

            if total_step % self.cfg.log_interval == 0:
                avg_return = sum(run_episode_reward) / len(run_episode_reward)
                if not self.cfg.silent:
                    print(f"Episode {ep} Step {total_step}: "
                          f"Avg return {avg_return:.2f}, "
                          f"Episode length {train_info['episode_length']}, logstd {logstd}")

                if self.cfg.save_logging:
                    train_info.update({'average_return': avg_return})
                    L.log(**train_info)
                run_episode_reward = []

        if self.cfg.save_model:
            self.save_model()
        logging_path = str(self.logging_dir) + '/logs'
        if self.cfg.save_logging:
            L.save(logging_path, self.seed)
        self.env.close()
        end = time.perf_counter()
        print("------ Training finished ------")
        print(f"Total training time: {(end - start) / 60:.2f} mins")

    def load_model(self):
        filepath = f'{self.model_dir}/model_parameters_{self.seed}.pt'
        state_dict = torch.load(filepath)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        print("Loaded model from", filepath)

    def save_model(self):
        filepath = f'{self.model_dir}/model_parameters_{self.seed}.pt'
        torch.save(self.policy.state_dict(), filepath)
        print("Saved model to", filepath)
