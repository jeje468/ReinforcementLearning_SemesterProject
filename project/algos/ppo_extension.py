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
        returns = []
        with torch.no_grad():
            _, values = self.policy(self.states)
            _, next_values = self.policy(self.next_states)
            values = values.squeeze()
            next_values = next_values.squeeze()
        gae = torch.tensor(0.0, device=self.device)
        timesteps = len(self.rewards)
        for t in range(timesteps-1, -1, -1):
            delta = self.rewards[t] + self.gamma * next_values[t] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.tau * (1 - self.dones[t]) * gae
            returns.append(gae + values[t])

        returns = list(reversed(returns))
        returns = torch.stack(returns).squeeze()
        return returns
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
        indices = list(range(len(self.states)))
        returns = self.compute_returns()
        while len(indices) >= self.batch_size:
            batch_indices = np.random.choice(indices, self.batch_size,
                    replace=False)

            self.ppo_update(self.states[batch_indices], self.actions[batch_indices],
                self.rewards[batch_indices], self.next_states[batch_indices],
                self.dones[batch_indices], self.action_log_probs[batch_indices],
                returns[batch_indices])

            indices = [i for i in indices if i not in batch_indices]
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
        c = 1.5
        action_dists, values = self.policy(states)
        values = values.squeeze()
        new_action_probs = action_dists.log_prob(actions)
        ratio = torch.exp(new_action_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)

        advantages = targets - values
        advantages -= advantages.mean()
        advantages /= advantages.std()+1e-8
        advantages = advantages.detach()
        L_clip = torch.min(ratio*advantages, clipped_ratio*advantages)
        neg_mask = (advantages < 0)
        L_dual = torch.where(neg_mask, torch.max(L_clip, c * advantages), L_clip)

        value_loss = F.smooth_l1_loss(values, targets, reduction="mean")

        L_dual = -(L_dual.mean())
        entropy = action_dists.entropy().mean()
        loss = L_dual + 0.5*value_loss - 0.01*entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ===== YOUR CODE ENDS HERE =====

    def get_action(self, observation, evaluation=False):
        """Select an action from the current policy."""
        # ===== YOUR CODE STARTS HERE =====
        # Convert observation to tensor, pass through policy,
        # sample or take mean action depending on evaluation flag.
        # Return both the action and its log probability.
        x = torch.from_numpy(observation).float().to(self.device)
        action_dist, _ = self.policy.forward(x)
        if evaluation:
            action = action_dist.mean.detach()
        else:
            action = action_dist.sample()
        aprob = action_dist.log_prob(action)
        return action, aprob
        # ===== YOUR CODE ENDS HERE =====

    def store_outcome(self, state, action, next_state, reward, action_log_prob, done):
        """Store one transition into the buffer."""
        # ===== YOUR CODE STARTS HERE =====
        # Append each element (as torch.Tensor) to self.states, self.actions, etc.
        self.states.append(torch.from_numpy(state).float())
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob.detach())
        self.rewards.append(torch.Tensor([reward]).float())
        self.dones.append(torch.Tensor([done]))
        self.next_states.append(torch.from_numpy(next_state).float())
        # ===== YOUR CODE ENDS HERE =====

    def train_iteration(self, ratio_of_episodes):
        """Run one environment episode and update policy when enough samples are collected."""
        # ===== YOUR CODE STARTS HERE =====
        # Steps:
        # 1. Reset the environment.
        # 2. Collect transitions until done or max steps.
        # 3. Call self.update_policy() periodically.
        # 4. Adjust policy exploration using self.policy.set_logstd_ratio().
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
        # ===== YOUR CODE ENDS HERE =====

    def train(self):
        """Overall training loop for multiple episodes."""
        # ===== YOUR CODE STARTS HERE =====
        # 1. Initialize logger if needed.
        # 2. Loop over training episodes, calling train_iteration().
        # 3. Track average returns, log results, and save models periodically.
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
        # ===== YOUR CODE ENDS HERE =====

    def load_model(self):
        """Load model weights."""
        # ===== YOUR CODE STARTS HERE =====
        filepath = f'{self.model_dir}/model_parameters_{self.seed}.pt'
        state_dict = torch.load(filepath)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()
        print("Loaded model from", filepath)
        # ===== YOUR CODE ENDS HERE =====

    def save_model(self):
        """Save model weights."""
        # ===== YOUR CODE STARTS HERE =====
        filepath = f'{self.model_dir}/model_parameters_{self.seed}.pt'
        torch.save(self.policy.state_dict(), filepath)
        print("Saved model to", filepath)
        # ===== YOUR CODE ENDS HERE =====