"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        hidden_dims: List[int] = [64, 64],
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        hidden_dims : List[int]
            List of sizes for each hidden layer in the Q-network.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q-network and frozen target
        self.q = QNetwork(obs_dim, n_actions, hidden_dims=hidden_dims)
        self.target_q = QNetwork(obs_dim, n_actions, hidden_dims=hidden_dims)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        return float(
            self.epsilon_final
            + (self.epsilon_start - self.epsilon_final)
            * np.exp(-self.total_steps / self.epsilon_decay)
        )

    def predict_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Choose action via ε-greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        """
        if evaluate:
            # TODO: select purely greedy action from Q(s)
            return self._get_best_action(state, evaluate)
        else:
            if np.random.rand() < self.epsilon():
                return self.env.action_space.sample()
            else:
                return self._get_best_action(state, evaluate)

    def _get_best_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        with torch.no_grad():
            qvals = self.q(torch.tensor(state, dtype=torch.float32))
            action = int(torch.argmax(qvals).item())
        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(
            *training_batch
        )  # noqa: F841
        s = torch.tensor(np.array(states), dtype=torch.float32)  # noqa: F841
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(
            1
        )  # noqa: F841
        r = torch.tensor(np.array(rewards), dtype=torch.float32)  # noqa: F841
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)  # noqa: F841
        mask = torch.tensor(np.array(dones), dtype=torch.float32)  # noqa: F841

        pred = self.q(s).gather(1, a)

        # Compute TD target with frozen network
        with torch.no_grad():
            q_next_target = self.target_q(s_next)
            max_q_next = q_next_target.max(1)[0]

            td_target_values = r + self.gamma * (1 - mask) * max_q_next

            # Reshape td_target_values to [batch_size, 1] to match pred's shape
            target = td_target_values.unsqueeze(1)

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(
        self, num_frames: int, eval_interval: int = 1000
    ) -> Tuple[List[int], List[float]]:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward. (Currently, logging is fixed at every 10 episodes)

        Returns
        -------
        frames_log : List[int]
            List of frame numbers at which average reward was logged.
        avg_rewards_log : List[float]
            List of average rewards logged.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []

        frames_log: List[int] = []
        avg_rewards_log: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) >= 10 and len(recent_rewards) % 10 == 0:
                    avg = np.mean(recent_rewards[-10:])
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )
                    frames_log.append(frame)
                    avg_rewards_log.append(avg)

        print("Training complete.")
        return frames_log, avg_rewards_log


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    import os
    import matplotlib.pyplot as plt

    experiment_configurations = [
        {
            "hidden_dims": [64, 64],
            "batch_size": 32,
            "buffer_capacity": 10000,
            "label": "h[64,64]_b32_c10k",
        },
        {
            "hidden_dims": [32],
            "batch_size": 32,
            "buffer_capacity": 10000,
            "label": "h[32]_b32_c10k",
        },
        {
            "hidden_dims": [128, 128],
            "batch_size": 32,
            "buffer_capacity": 10000,
            "label": "h[128,128]_b32_c10k",
        },
        {
            "hidden_dims": [64, 64, 64],
            "batch_size": 32,
            "buffer_capacity": 10000,
            "label": "h[64,64,64]_b32_c10k",
        },
        {
            "hidden_dims": [64, 64],
            "batch_size": 16,
            "buffer_capacity": 10000,
            "label": "h[64,64]_b16_c10k",
        },
        {
            "hidden_dims": [64, 64],
            "batch_size": 64,
            "buffer_capacity": 10000,
            "label": "h[64,64]_b64_c10k",
        },
        {
            "hidden_dims": [64, 64],
            "batch_size": 32,
            "buffer_capacity": 5000,
            "label": "h[64,64]_b32_c5k",
        },
        {
            "hidden_dims": [64, 64],
            "batch_size": 32,
            "buffer_capacity": 20000,
            "label": "h[64,64]_b32_c20k",
        },
        {
            "hidden_dims": [32],
            "batch_size": 16,
            "buffer_capacity": 5000,
            "label": "h[32]_b16_c5k",
        },
        {
            "hidden_dims": [128, 128],
            "batch_size": 64,
            "buffer_capacity": 20000,
            "label": "h[128,128]_b64_c20k",
        },
    ]

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    env_name_sanitized = cfg.env.name.lower().replace("-", "_").replace("/", "_")

    for exp_config in experiment_configurations:
        hidden_dims = exp_config["hidden_dims"]
        batch_size = exp_config["batch_size"]
        buffer_capacity = exp_config["buffer_capacity"]
        config_label = exp_config["label"]

        print(
            f"\nTraining with config: {config_label} (HiddenDims: {hidden_dims}, BatchSize: {batch_size}, BufferCap: {buffer_capacity})"
        )

        env = gym.make(cfg.env.name)

        agent = DQNAgent(
            env,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            lr=cfg.agent.learning_rate,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            target_update_freq=cfg.agent.target_update_freq,
            seed=cfg.seed,
            hidden_dims=hidden_dims,
        )
        frames_history, avg_rewards_history = agent.train(
            num_frames=cfg.train.num_frames,
            eval_interval=cfg.train.eval_interval,
        )

        plt.figure()
        plt.plot(frames_history, avg_rewards_history)
        plt.xlabel("Frames")
        plt.ylabel("Average Reward (Last 10 Episodes)")

        plot_title = f"DQN ({config_label}) on {cfg.env.name}"
        plt.title(plot_title)

        plot_filename = os.path.join(
            plot_dir, f"dqn_{env_name_sanitized}_{config_label}_training_curve.png"
        )

        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        plt.close()


if __name__ == "__main__":
    main()
