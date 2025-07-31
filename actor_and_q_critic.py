import collections
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from quanser_env import QuanserEnv

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, n_features: int = 5, n_actions: int = 1):
        super().__init__()
        self.n_actions = n_actions
        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)
        self.epsilon = None
        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        elif isinstance(x, torch.Tensor):
            x = x.to(DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu_v = F.tanh(self.out(x))

        return mu_v

    def get_action(self, x: torch.Tensor, scale: float = 1.0, exploration: bool = True) -> np.ndarray:
        mu_v = self.forward(x)

        action = mu_v.detach().cpu().numpy()

        if exploration:
            noises = np.random.normal(size=self.n_actions, loc=0, scale=scale)
            action = action + noises

        action = np.clip(action, a_min=-1., a_max=1.)

        return action

    def get_action_e(self, env: QuanserEnv, x: torch.Tensor, step: int, epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: int = 10000) -> np.ndarray:
        self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, (epsilon_decay - step) / epsilon_decay)

        if random.random() < self.epsilon:
            action = np.array(env.action_space.sample())
        else:
            mu_v = self.forward(x)

            action = mu_v.detach().numpy()
        action = np.clip(action, a_min=-1., a_max=1.)

        return action


class QCritic(nn.Module):
    """
    Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
    update. This a Neural Net with 1 hidden layer
    """

    def __init__(self, n_features: int = 5, n_actions: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(n_features + n_actions, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.to(DEVICE)

    def forward(self, x, action) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

Transition = collections.namedtuple(
    typename="Transition", field_names=["observation", "action", "next_observation", "reward", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def pop(self) -> Transition:
        return self.buffer.pop()

    def clear(self) -> None:
        self.buffer.clear()

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get random index
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        # Sample
        observations, actions, next_observations, rewards, dones = zip(*[self.buffer[idx] for idx in indices])

        # Convert to ndarray for speed up cuda
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        # observations.shape, next_observations.shape: (32, 5), (32, 5)

        actions = np.array(actions)
        actions = np.expand_dims(actions, axis=-1) if actions.ndim == 1 else actions
        rewards = np.array(rewards)
        rewards = np.expand_dims(rewards, axis=-1) if rewards.ndim == 1 else rewards
        dones = np.array(dones, dtype=bool)
        # actions.shape, rewards.shape, dones.shape: (32, 1) (32, 1) (32,)

        # Convert to tensor
        observations = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones
