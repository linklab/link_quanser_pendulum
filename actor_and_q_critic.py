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
# DEVICE = torch.device("cpu")

def init_weights_pendulum(m: nn.Module):
    if isinstance(m, nn.Linear):
        # (1) 은닉층: He(kaiming) 정규분포 + gain=√2 (ReLU용)
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, n_features: int = 5, n_actions: int = 1):
        super().__init__()
        self.n_actions = n_actions

        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_actions)

        self.apply(init_weights_pendulum)

        eps = 3e-3
        self.out.weight.data.uniform_(-eps, eps)
        self.out.bias.data.fill_(0.0)

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

    def get_action(self, x: torch.Tensor, scale: float = 1.0, exploration: bool = True, gym_pendulum: bool = False) -> np.ndarray:
        if gym_pendulum:
            mu_v = self.forward(x) * 2.0
        else:
            mu_v = self.forward(x).squeeze(dim=-1)

        action = mu_v.detach().cpu().numpy()

        if exploration:
            noise = np.random.normal(size=self.n_actions, loc=0.0, scale=scale)
            action = action + noise

        if gym_pendulum:
            action = np.clip(action, a_min=-2.0, a_max=2.0)
        else:
            action = np.clip(action, a_min=-1.0, a_max=1.0)

        return action

    def get_action_e(self, env: QuanserEnv, x: torch.Tensor, step: int, epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: int = 10000) -> np.ndarray:
        self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, (epsilon_decay - step) / epsilon_decay)

        if random.random() < self.epsilon:
            action = np.array(env.action_space.sample())
        else:
            mu_v = self.forward(x)

            action = mu_v.detach().numpy()
        action = np.clip(action, a_min=-1., a_max=1.)

        return action, self.epsilon


class QCritic(nn.Module):
    """
    Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
    update. This a Neural Net with 1 hidden layer
    """

    def __init__(self, n_features: int = 5, n_actions: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 256)
        # self.ln1 = nn.LayerNorm(256)
        #self.fc2 = nn.Linear(256 + n_actions, 256)  # for TD3
        self.fc2 = nn.Linear(256, 256)  # for DQN
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        #self.fc5 = nn.Linear(256, 1)  # for TD3: 1 output
        self.fc5 = nn.Linear(256, n_actions)  # for DQN: 3 actions

        self.apply(init_weights_pendulum)

        self.to(DEVICE)

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        elif isinstance(x, torch.Tensor):
            x = x.to(DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.fc5(x)
        return x
    
    def get_action_e(
            self,
            env: QuanserEnv,
            x: torch.Tensor,
            step: int,
            exploration: bool = True,
            gym_pendulum: bool  = False,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay: int = 10000
    ) -> np.ndarray:
        # if gym_pendulum:
        #     mu_v = self.forward(x) * 2.0
        # else:
        mu_v = self.forward(x).squeeze(dim=-1)
        q_values = mu_v.detach().cpu().numpy()
        action = int(np.argmax(q_values))

        if exploration:
            self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0, (epsilon_decay - step) / epsilon_decay)

            if random.random() < self.epsilon:
                # rand_q_values = np.array(env.dqn_action_space.sample())
                rand_q_value = random.randrange(0, 2)
                return rand_q_value, self.epsilon
            else:
                return action, self.epsilon
        else:
            return action, epsilon_end
        
    def get_action_eps(
            self,
            x: torch.Tensor,
            epsilon: float = 0.1,
    ) -> np.ndarray:

        if random.random() < epsilon:
            # rand_q_values = np.array(env.dqn_action_space.sample())
            action = random.randrange(0, 5)
        else:
            q_values = self.forward(x)
            action = torch.argmax(q_values, dim=-1)
            action = action.item()
        return action

     
    

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
        actions = torch.tensor(actions, dtype=torch.int64, device=DEVICE)  # TD3: float32, DQN: int64
        next_observations = torch.tensor(next_observations, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

        return observations, actions, next_observations, rewards, dones


class PER:
    def __init__(self, capacity: int, alpha: float = 0.6, 
                 beta_start: float = 0.4, beta_end: float = 1.0, beta_decay: int = 100_000):
        # PER hyperparameters
        self.alpha = alpha                              # 우선순위 샘플링 정도 (α)
        self.beta = beta_start                          # 중요도 가중치 초기값 (β 시작값)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_decay = beta_decay                    # β가 1에 도달하는 예상 경과 step 수
        self.buffer = collections.deque(maxlen=capacity)
        self.priorities = collections.deque(maxlen=capacity)
        self.max_priority = 1.0                         # 현재 우선순위 최대값 (새 샘플 추가 시 사용)

    def size(self) -> int:
        return len(self.buffer)

    def append(self, transition) -> None:
        """새로운 경험 추가: 우선순위를 현재 max_priority로 설정"""
        self.buffer.append(transition)
        # 버퍼에 추가되면 priorities도 동일하게 추가 (deque가 가득차면 가장 오래된 것 자동 제거)
        self.priorities.append(self.max_priority)  # 새로운 transition은 최대 우선순위로 부여

    def sample(self, batch_size: int):
        """우선순위 기반 샘플링하여 배치 반환 (순위 기반 확률, IS 가중치 계산 포함)"""
        assert len(self.buffer) >= batch_size, "Not enough samples to draw"
        N = len(self.buffer)
        # Priorities 배열 및 순위 계산
        priorities = np.array(self.priorities, dtype=np.float64)
        # 우선순위 큰 순으로 정렬하여 rank 산정 (rank 1 = highest priority)
        sorted_indices = np.argsort(priorities)[::-1]           # 우선순위 내림차순 인덱스 배열
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, N+1)               # 각 인덱스의 rank (1~N)
        # Rank 기반 확률분포 P(i) 계산
        probs = 1.0 / (ranks ** self.alpha)
        probs /= probs.sum()                                   # 정규화하여 합=1 확률분포
        # 중요도 샘플링 β 값 선형 증가 (annealing)
        self.beta = min(self.beta_end, self.beta + (self.beta_end - self.beta_start) / self.beta_decay)
        # 우선순위 확률분포에 따라 샘플 인덱스 뽑기 (Not Random!!!)
        indices = np.random.choice(N, size=batch_size, p=probs, replace=False)
        # 인덱스에 해당하는 transition 묶기
        batch = [self.buffer[idx] for idx in indices]
        observations, actions, next_observations, rewards, dones = zip(*batch)
        # NumPy 배열 -> Tensor 변환 (장치 이동 포함)
        observations = torch.tensor(np.array(observations), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=DEVICE).unsqueeze(-1)
        next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        dones = torch.tensor(np.array(dones, dtype=bool), dtype=torch.bool, device=DEVICE)
        # 중요도 샘플링 가중치 계산
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()    # max weight를 1로 정규화
        weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        # 샘플된 배치의 rank (우선순위 크기 비교용)
        sampled_ranks = torch.tensor(ranks[indices], dtype=torch.float32, device=DEVICE)

        return indices, observations, actions, next_observations, rewards, dones, weights, sampled_ranks
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """학습 후 배치 샘플들의 TD 오차로 우선순위 업데이트"""
        for idx, err in zip(indices, td_errors):
            # 새로운 우선순위 값을 TD 오차 크기로 설정 (ε 추가하여 0 회피)
            new_priority = float(abs(err)) + 1e-6
            self.priorities[idx] = new_priority
            # max_priority 갱신
            if new_priority > self.max_priority:
                self.max_priority = new_priority
