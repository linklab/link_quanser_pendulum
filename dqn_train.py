from collections import deque
from quanser.hardware import HIL
from array import array
import numpy as np
import gymnasium as gym
import threading

card = HIL("qube_servo3_usb", "0")
led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

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
from actor_and_q_critic import DEVICE, MODEL_DIR, QCritic, ReplayBuffer, Transition, PER

import wandb
np.set_printoptions(precision=5, suppress=True)

DISCRETE_PWMS = [-0.35, -0.02, 0.0, 0.02, 0.35]  # 3 actions


class DQN:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        custom_name = "DQN"
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
        self.training_start_steps = config["training_start_steps"]
        self.alpha = config["alpha"]
        self.beta_start = config["beta_start"]
        self.beta_end = config["beta_end"]
        self.beta_decay = config["beta_decay"]
        self.epsilon_final_scehduled_percent = config["epsilon_final_scehduled_percent"]
        self.target_sync_time_steps_interval = config["target_sync_time_steps_interval"]
        self.validation_time_steps_interval = config["validation_time_steps_interval"]

        self.q_critic = QCritic(n_features=6, n_actions=5)
        self.target_q_critic = QCritic(n_features=6, n_actions=5)
        self.target_q_critic.load_state_dict(self.q_critic.state_dict())
        self.q_critic_optimizer = optim.Adam(self.q_critic.parameters(), lr=self.learning_rate)


        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        self.epsilon_scheduled_last_episode = self.max_num_episodes * self.epsilon_final_scehduled_percent

        self.time_steps = 0
        self.training_time_steps = 0
        self.epsilon_steps = 0


        self.total_train_start_time = None
        self.train_start_time = None
        self.step_call_time = None
  

    def epsilon_scheduled(self, current_episode: int) -> float:
        if current_episode < 25:
            return self.epsilon_start
        fraction = min((current_episode - 25) / self.epsilon_scheduled_last_episode, 1.0)
        epsilon_span = self.epsilon_start - self.epsilon_end

        epsilon = min(self.epsilon_start - fraction * epsilon_span, self.epsilon_start)
        return epsilon

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()
        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            loss_list = []
            episode_reward = 0.0
            episode_steps = 0
            done = False
            step_call_time_deque = deque(maxlen=2000)
            action_list = deque(maxlen=2000)
            self.step_call_time = None
            epsilon = self.epsilon_scheduled(n_episode)
            held_action = None

            observation = self.env.reset()  # motor_ang, sin(pend_ang), cos(pend_ang), motor_ang_vel, pend_ang_vel, pend_spin_num
            print("======EPISODE START====== ")
            episode_start_time = time.time()
            while not done:
                self.train_start_time = time.time()
                self.time_steps += 1

                self.epsilon_steps += 1
                if n_episode < 25:
                    action = self.q_critic.get_action_eps(observation, 1.0) # random action
                else:

                    action = self.q_critic.get_action_eps(observation, epsilon) # epsilon-greedy

                action_list.append(DISCRETE_PWMS[action])

                self.env.apply_action_dqn(DISCRETE_PWMS[action])

                if self.time_steps > self.training_start_steps:
                    for _ in range(2):
                        loss = self.train()
                        loss_list.append(loss)
                
                if self.step_call_time is not None:
                    while (time.perf_counter() - self.step_call_time) < 0.006:
                        time.sleep(0.0001)
                    step_call_time_deque.append(time.perf_counter() - self.step_call_time - 0.006)
                    
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                self.step_call_time = time.perf_counter()

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                episode_steps += 1
                

            print("======EPISODE END====== ")
            episode_time = time.time() - episode_start_time
            if len(step_call_time_deque) > 0:
                step_call_time_deque.pop()
                print(f"STEP CALL TIME ERROR MEAN: {np.mean(step_call_time_deque):.6f} sec")


            if n_episode % 10 == 0 and n_episode > 50:
                from array import array
                green_led_values = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                card.write_other(led_channels, len(led_channels), green_led_values)
                is_terminated = self.validate()

            loss_mean = np.mean(loss_list)


            if n_episode % self.print_episode_interval == 0:
                print(
                    "\n[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "\nEpisode Reward: {:>9.3f},".format(episode_reward),
                    "\nLoss: {:>7.3f},".format(loss_mean),
                    "\nTraining Steps: {:5,}, ".format(self.training_time_steps),
                    "\nEpisode Time: {},".format(time.strftime("%H:%M:%S", time.gmtime(episode_time))),
                    "\nEpsilon: {:>7.3f}, \n".format(epsilon),
                )
                print("####################################################################################")

            if self.use_wandb:
                self.log_wandb(
                    episode_reward,
                    loss_mean,
                    n_episode,
                    epsilon,
                    action_list,
                    self.time_steps
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
        loss: float,
        n_episode: float,
        epsilon: float,
        action_list: deque,
        total_steps: int
    ) -> None:
        self.wandb.log(
            {
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Loss": loss,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "[TRAIN] Epsilon": epsilon,
                "[TRAIN] EPISODE ACTION MEAN": np.mean(action_list),
                "[TRAIN] Total Steps": total_steps,
                "Training Episode": n_episode,
                "Training Steps": self.training_time_steps,
            }
        )
    
    def train(self) -> tuple[float, float, float]:
        self.training_time_steps += 1
        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)


        q_values = self.q_critic(observations) # (batch, n_actions)
        q_value = q_values.gather(dim=1, index=actions) # (batch, 1)
        with torch.no_grad():
            # DQN
            q_values_next = self.target_q_critic(next_observations)  # (batch, n_actions)
            max_vals= q_values_next.max(dim=1, keepdim=True).values  # (batch, 1)
            max_vals[dones] = 0.0
            target = rewards + self.gamma * max_vals  # (batch, 1)



        
        loss = F.mse_loss(q_value, target)  # (batch, 1)



        self.q_critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_critic.parameters(), 1.0)
        self.q_critic_optimizer.step()


        self.soft_synchronize_models(source_model=self.q_critic, target_model=self.target_q_critic, tau=self.soft_update_tau)




        return loss.item()
    


    def soft_synchronize_models(self, source_model, target_model, tau) -> None:
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = (1.0 - tau) * target_model_state[k] + tau * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, val_reward_avg: float) -> None:
        # check model directory path
        os.makedirs(MODEL_DIR, exist_ok=True)

        # save actor
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"dqn_{self.env_name}_{val_reward_avg:.1f}_qnet_{timestamp}.pth"
        filepath  = os.path.join(MODEL_DIR, filename)

        torch.save(self.q_critic.state_dict(), filepath)

        copyfile(filepath, os.path.join(MODEL_DIR, f"dqn_{self.env_name}_qnet_latest.pth"))


    def validate(self) -> bool:
        is_terminated = False
        validation_episode_reward_list = []

        for i in range(3):
            validation_episode_reward = 0
            observation = self.env.reset()
            done = False
            print("***********VALIDATION START*********** ")
            while not done:

                action = self.q_critic.get_action_eps(observation, 0.0)

                self.env.apply_action_dqn(DISCRETE_PWMS[action])

                time.sleep(0.006)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                validation_episode_reward += reward
                observation = next_observation
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
        "max_num_episodes": 1500,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "replay_buffer_size": 200_000,                    # 리플레이 버퍼 사이즈
        "learning_rate": 0.0001,                            # 학습율
        "gamma": 0.999,                                      # 감가율
        "soft_update_tau": 0.005,                           # td3 Soft Update Tau
        "print_episode_interval": 1,                        # Episode 통계 출력에 관한 에피소드 간격
        "episode_reward_avg_solved": 3950.0,                   # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        "epsilon_start": 0.95,
        "epsilon_end": 0.1,
        "training_start_steps": 256,
        "seed": 42,
        "alpha": 0.75,
        "beta_start": 0.0,
        "beta_end": 1.0,
        "beta_decay": 500_000,
        "epsilon_final_scehduled_percent": 0.06,
        "target_sync_time_steps_interval": 2000,
        "validation_time_steps_interval": 5000,
    }
    seed_everything(config["seed"])

    print(env.observation_space)
    print(env.dqn_action_space)
    use_wandb = True
    dqn = DQN(env=env, test_env=env, config=config, use_wandb=use_wandb)
    dqn.train_loop()

if __name__ == "__main__":
    main()