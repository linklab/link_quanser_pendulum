from collections import deque
from quanser.hardware import HIL
from array   import array
import numpy as np
card = HIL("qube_servo3_usb", "0")
led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)
import gymnasium as gym

from quanser_env import QuanserEnv

# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import sys
import os
import time
from datetime import datetime
from shutil import copyfile

import random
import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils as tnn_utils
import torch.optim as optim
from actor_and_q_critic import DEVICE, MODEL_DIR, Actor, QCritic, ReplayBuffer, Transition

import wandb
np.set_printoptions(precision=5, suppress=True)

class LinearNoiseScheduler:
    def __init__(self, start_scale: float, end_scale: float, decay_steps: int):
        """
        :param start_scale: 초기 노이즈 크기
        :param end_scale: 최종 노이즈 크기
        :param decay_steps: start→end로 감쇠시킬 총 스텝 수
        """
        self.start = start_scale
        self.end = end_scale
        self.decay_steps = decay_steps

    def get_scale(self, step: int) -> float:
        """현재 step에 대응하는 scale 값을 반환."""
        if step >= self.decay_steps:
            return self.end
        # 선형 보간
        frac = step / self.decay_steps
        return self.start + frac * (self.end - self.start)


class TD3:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        custom_name = "td3"
        self.current_time = custom_name + datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_")

        if use_wandb:
            self.wandb = wandb.init(project="DDPG_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        self.training_start_steps = config["training_start_steps"]

        self.actor = Actor(n_features=5, n_actions=1)  # n_features: (motor_angle, pendulum_angle, motor_ang_vel, pendulum_ang_vel, last_action), shape: (5,)
        self.target_actor = Actor(n_features=5, n_actions=1)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.q_critic_1 = QCritic(n_features=5, n_actions=1)
        self.target_q_critic_1 = QCritic(n_features=5, n_actions=1)
        self.target_q_critic_1.load_state_dict(self.q_critic_1.state_dict())
        self.q_critic_1_optimizer = optim.Adam(self.q_critic_1.parameters(), lr=self.learning_rate)

        self.q_critic_2 = QCritic(n_features=5, n_actions=1)
        self.target_q_critic_2 = QCritic(n_features=5, n_actions=1)
        self.target_q_critic_2.load_state_dict(self.q_critic_2.state_dict())
        self.q_critic_2_optimizer = optim.Adam(self.q_critic_2.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        self.time_steps = 0
        self.training_time_steps = 0


        self.total_train_start_time = None
        self.train_start_time = None

        self.step_call_time = None

        self.nosie_scale_scheduler = LinearNoiseScheduler(start_scale=1.0, end_scale=0.1, decay_steps=100_000)

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()

        policy_loss = critic_loss = mu_v = 0.0

        is_terminated = False
        for n_episode in range(1, self.max_num_episodes + 1):
            policy_loss_list = []
            critic_loss_list = []
            mu_v_list = []
            episode_reward = 0
            episode_steps = 0
            done = False
            step_call_time_deque = deque(maxlen=2000)
            action_list = deque(maxlen=2000)
            self.step_call_time = None

            observation = self.env.reset()
            print("======EPISODE START====== ")
            while not done:
                self.train_start_time = time.time()
                self.time_steps += 1
                if n_episode < 50:
                    scale = 2.0
                else:
                    scale = self.nosie_scale_scheduler.get_scale(self.time_steps)
                action = self.actor.get_action(observation, scale=scale, exploration=False)
                if episode_steps % 5 == 0:
                    noise = np.random.normal(size=1, loc=0.0, scale=scale)
                action += noise
                action = np.clip(action, a_min=-1.0, a_max=1.0)
                action_list.append(action.item())

                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                if self.step_call_time is not None:
                    step_call_time_error = time.perf_counter() - self.step_call_time
                    step_call_time_deque.append(step_call_time_error - 0.006)
                self.step_call_time = time.perf_counter()

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated
                
                episode_steps += 1

            print("======EPISODE END====== ")
            step_call_time_deque.pop()
            # print(f"STEP CALL TIME DEQUE: {np.asarray(step_call_time_deque)}")
            print(f"STEP CALL TIME ERROR MEAN: {np.mean(step_call_time_deque):.6f} sec")

            if episode_reward > self.episode_reward_avg_solved - 300.0:
                is_terminated = self.validate()

            if self.time_steps >= self.training_start_steps:
                iter_training_steps = np.max([episode_steps, 500])
                for _ in range(iter_training_steps):
                    policy_loss, critic_loss, mu_v = self.train()
                    if policy_loss is not None:
                        policy_loss_list.append(policy_loss)
                        mu_v_list.append(mu_v)
                    critic_loss_list.append(critic_loss)

            if n_episode % 10 == 0 and n_episode > 100:
                from array import array
                green_led_values = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                card.write_other(led_channels, len(led_channels), green_led_values)
                is_terminated = self.validate()


            policy_loss_mean = np.mean(policy_loss_list)
            critic_loss_mean = np.mean(critic_loss_list)
            mu_v_mean = np.mean(mu_v_list)

            if n_episode % self.print_episode_interval == 0:
                print(
                    "\n[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "\nEpisode Reward: {:>9.3f},".format(episode_reward),
                    "\nmu_v Loss: {:>7.3f},".format(policy_loss_mean),
                    "\nCritic Loss: {:>7.3f},".format(critic_loss_mean),
                    "\nTraining Steps: {:5,}, ".format(self.training_time_steps),
                    "\nTraining Time: {}, \n".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - self.train_start_time))),
                )
                print("####################################################################################")

            if self.use_wandb:
                self.log_wandb(
                    episode_reward,
                    policy_loss_mean,
                    critic_loss_mean,
                    mu_v_mean,
                    n_episode,
                    scale,
                    action_list
                )

            if is_terminated:
                break

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(
        self,
        episode_reward: float,
        policy_loss: float,
        critic_loss: float,
        mu_v: float,
        n_episode: float,
        scale: float,
        action_list: deque
    ) -> None:
        self.wandb.log(
            {
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Policy Loss": policy_loss,
                "[TRAIN] Critic Loss": critic_loss,
                "[TRAIN] Batch mu_v": mu_v,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "[TRAIN] Noise Scale": scale,
                "[TRAIN] EPISODE ACTION MEAN": np.mean(action_list),
                "Training Episode": n_episode,
                "Training Steps": self.training_time_steps,
            }
        )

    def train(self) -> tuple[float, float, float]:
        self.training_time_steps += 1
        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        # CRITIC UPDATE
        q_values_1 = self.q_critic_1(observations, actions).squeeze(dim=-1)
        q_values_2 = self.q_critic_2(observations, actions).squeeze(dim=-1)
        with torch.no_grad():
            # target policy smoothing
            next_mu_v = self.target_actor(next_observations)
            noise = (torch.randn_like(next_mu_v) * 0.2).clamp_(-0.5, 0.5)
            next_mu_v = (next_mu_v + noise).clamp_(-1.0, 1.0)

            next_target_q1 = self.target_q_critic_1(next_observations, next_mu_v).squeeze(dim=-1)
            next_target_q2 = self.target_q_critic_2(next_observations, next_mu_v).squeeze(dim=-1)

            # prevent overestimate Q-values
            next_target_q_min = torch.minimum(next_target_q1, next_target_q2)

            next_target_q_min[dones] = 0.0

            target_values = rewards.squeeze(dim=-1) + self.gamma * next_target_q_min

        critic_loss_1 = F.mse_loss(target_values, q_values_1)
        self.q_critic_1_optimizer.zero_grad()
        critic_loss_1.backward()
        tnn_utils.clip_grad_norm_(self.q_critic_1.parameters(), max_norm=1.0)
        self.q_critic_1_optimizer.step()
        self.soft_synchronize_models(
            source_model=self.q_critic_1, target_model=self.target_q_critic_1, tau=self.soft_update_tau
        )

        critic_loss_2 = F.mse_loss(target_values, q_values_2)
        self.q_critic_2_optimizer.zero_grad()
        critic_loss_2.backward()
        tnn_utils.clip_grad_norm_(self.q_critic_2.parameters(), max_norm=1.0)
        self.q_critic_2_optimizer.step()
        self.soft_synchronize_models(
            source_model=self.q_critic_2, target_model=self.target_q_critic_2, tau=self.soft_update_tau
        )

        critic_loss = (critic_loss_1 + critic_loss_2) / 2.0

        # delayed policy update 
        if self.training_time_steps  % 2 == 0:
            for p in self.q_critic_1.parameters():  # grad 차단
                p.requires_grad = False

            # ACTOR UPDATE
            mu_v = self.actor(observations)
            q_v = self.q_critic_1(observations, mu_v)
            actor_objective = q_v.mean()
            actor_loss = -1.0 * actor_objective

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            tnn_utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            for p in self.q_critic_1.parameters():  # grad 차단
                p.requires_grad = True

            # sync, TAU: 0.995
            self.soft_synchronize_models(
                source_model=self.actor, target_model=self.target_actor, tau=self.soft_update_tau
            )
            return actor_loss.item(), critic_loss.item(), mu_v.mean().item()
        else:
            return None, critic_loss.item(), None

    def soft_synchronize_models(self, source_model, target_model, tau) -> None:
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = tau * target_model_state[k] + (1.0 - tau) * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, val_reward_avg: float) -> None:
        # check model directory path
        os.makedirs(MODEL_DIR, exist_ok=True)

        # save actor
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"td3_{self.env_name}_{val_reward_avg:.1f}_actor_{timestamp}.pth"
        filepath  = os.path.join(MODEL_DIR, filename)

        torch.save(self.actor.state_dict(), filepath)

        copyfile(filepath, os.path.join(MODEL_DIR, f"td3_{self.env_name}_actor_latest.pth"))

        # save critic 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"td3_{self.env_name}_{val_reward_avg:.1f}_critic_1_{timestamp}.pth"
        filepath  = os.path.join(MODEL_DIR, filename)

        torch.save(self.q_critic_1.state_dict(), filepath)

        copyfile(filepath, os.path.join(MODEL_DIR, f"td3_{self.env_name}_critic_1_latest.pth"))

        # save critic 2
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"td3_{self.env_name}_{val_reward_avg:.1f}_critic_2_{timestamp}.pth"
        filepath  = os.path.join(MODEL_DIR, filename)

        torch.save(self.q_critic_2.state_dict(), filepath)

        copyfile(filepath, os.path.join(MODEL_DIR, f"td3_{self.env_name}_critic_2_latest.pth"))

    def validate(self) -> bool:
        is_terminated = False
        validation_episode_reward_list = []

        for i in range(3):
            validation_episode_reward = 0
            observation = self.env.reset()
            done = False
            print("***********VALIDATION START*********** ")
            while not done:
                action = self.actor.get_action(observation, exploration=False)

                observation, reward, terminated, truncated, _ = self.env.step(action)

                validation_episode_reward += reward

                done = terminated or truncated

            validation_episode_reward_list.append(validation_episode_reward)
            print(f"Validation {i+1} reward: {validation_episode_reward}")
        validation_episode_reward_mean = np.mean(validation_episode_reward_list)
        print(f"[Validation Episode Reward Mean: {validation_episode_reward_mean}]")
        if self.use_wandb:
            self.wandb.log({"[VALIDATION] Episode Reward": validation_episode_reward_mean})

        if validation_episode_reward_mean > self.episode_reward_avg_solved:
            print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
            self.model_save(validation_episode_reward_mean)
            is_terminated = True
        return is_terminated
        

def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)     # 파이썬 해시 시드
    random.seed(seed)                            # random 모듈
    np.random.seed(seed)                         # NumPy
    torch.manual_seed(seed)                      # CPU
    torch.cuda.manual_seed_all(seed)             # 모든 GPU
    torch.backends.cudnn.deterministic = True    # 알고리즘 고정
    torch.backends.cudnn.benchmark = False       # 튜닝 비활성화

def main():
    env = QuanserEnv(card)

    config = {
        "env_name": "quanser",                              # 환경의 이름
        "max_num_episodes": 200_000,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "replay_buffer_size": 100_000,                    # 리플레이 버퍼 사이즈
        "learning_rate": 0.0005,                            # 학습율
        "gamma": 0.999,                                      # 감가율
        "soft_update_tau": 0.998,                           # td3 Soft Update Tau
        "print_episode_interval": 1,                        # Episode 통계 출력에 관한 에피소드 간격
        "episode_reward_avg_solved": 3950.0,                   # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 100_000,
        "training_start_steps": 256,
        "seed": 42
    }
    seed_everything(config["seed"])

    print(env.observation_space)
    print(env.action_space)
    use_wandb = True
    td3 = TD3(env=env, test_env=env, config=config, use_wandb=use_wandb)
    td3.train_loop()

if __name__ == "__main__":
    main()