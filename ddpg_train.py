from quanser.hardware import HIL
from array   import array
import gymnasium as gym

from quanser_env import QuanserEnv

# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from actor_and_q_critic import MODEL_DIR, Actor, QCritic, ReplayBuffer, Transition

import wandb


class DDPG:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="DDPG_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.validation_time_steps_interval = config["validation_time_steps_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.steps_between_train = config["steps_between_train"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]

        self.actor = Actor(n_features=5, n_actions=1)  # n_features: (motor_angle, pendulum_angle, motor_ang_vel, pendulum_ang_vel, last_action), shape: (5,)
        self.target_actor = Actor(n_features=5, n_actions=1)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        self.q_critic = QCritic(n_features=5, n_actions=1)
        self.target_q_critic = QCritic(n_features=5, n_actions=1)
        self.target_q_critic.load_state_dict(self.q_critic.state_dict())
        self.q_critic_optimizer = optim.Adam(self.q_critic.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        self.time_steps = 0
        self.training_time_steps = 0

        self.best_count = 0 # 설정한 목표에 도달한 횟수
        self.best_saved = False

        self.total_train_start_time = None

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()

        # validation_episode_reward_avg = -1500
        policy_loss = critic_loss = mu_v = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0

            self.env.reset()
            observation = self.env.get_init_observations()
            print("init_obs: ", observation)
            done = False

            print("======EPISODE START====== ")
            while not done:
                self.time_steps += 1
                # batch_size = 32
                # action = torch.empty(batch_size).uniform_(-1.0, 1.0) # batch random
                # action = torch.empty(1).uniform_(-1.0, 1.0)
                # action = torch.zeros(1)
                action = self.actor.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                # if self.time_steps % 10 == 0:
                    # print(f"random action: {action}")
                    # print("===================================================================")
                    # print(f"next_obs: {next_observation}")
                    # print(f"reward: {reward:.2f}")
                    # print(f"\nterminated: {terminated}")
                    # print(f"truncated: {truncated}")
                    # print(f"info: {info}")
                    # print("===================================================================")
                    # print()
                # if self.env.step_count % 200 == 0:
                #     print("======EPISODE END====== ")
                #     break
                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    policy_loss, critic_loss, mu_v = self.train()

                # if self.time_steps % self.validation_time_steps_interval == 0:
                #     validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                if not self.best_saved and episode_reward > self.episode_reward_avg_solved:
                    self.best_count += 1
                    if self.best_count >= 10: # 10번 설정한 목표에 도달하면 학습 종료
                        print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                        self.model_save(episode_reward)
                        is_terminated = True
                        self.best_saved = True

            print("======EPISODE END====== ")

            if n_episode % self.print_episode_interval == 0:
                print("####################################################################################")
                print(
                    "[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>9.3f},".format(episode_reward),
                    "Policy Loss: {:>7.3f},".format(policy_loss),
                    "Critic Loss: {:>7.3f},".format(critic_loss),
                    "Training Steps: {:5,}, ".format(self.training_time_steps),
                )
                print("####################################################################################")

            # if self.use_wandb and n_episode > self.validation_time_steps_interval:
            if self.use_wandb:
                self.log_wandb(
                    episode_reward,
                    policy_loss,
                    critic_loss,
                    mu_v,
                    n_episode,
                )

            if is_terminated:
                if self.wandb:
                    for _ in range(5):
                        self.log_wandb(
                            episode_reward,
                            policy_loss,
                            critic_loss,
                            mu_v,
                            n_episode,
                        )
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
    ) -> None:
        self.wandb.log(
            {
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Policy Loss": policy_loss,
                "[TRAIN] Critic Loss": critic_loss,
                "[TRAIN] mu_v": mu_v,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "Training Episode": n_episode,
                "Training Steps": self.training_time_steps,
            }
        )

    def train(self) -> tuple[float, float, float]:
        self.training_time_steps += 1

        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        # CRITIC UPDATE
        q_values = self.q_critic(observations, actions).squeeze(dim=-1)
        next_mu_v = self.target_actor(next_observations)
        next_q_values = self.target_q_critic(next_observations, next_mu_v).squeeze(dim=-1)
        next_q_values[dones] = 0.0
        target_values = rewards.squeeze(dim=-1) + self.gamma * next_q_values
        critic_loss = F.mse_loss(target_values.detach(), q_values)
        self.q_critic_optimizer.zero_grad()
        critic_loss.backward()
        self.q_critic_optimizer.step()

        # ACTOR UPDATE
        mu_v = self.actor(observations)
        q_v = self.q_critic(observations, mu_v)
        actor_objective = q_v.mean()
        actor_loss = -1.0 * actor_objective

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # sync, TAU: 0.995
        self.soft_synchronize_models(
            source_model=self.actor, target_model=self.target_actor, tau=self.soft_update_tau
        )
        self.soft_synchronize_models(
            source_model=self.q_critic, target_model=self.target_q_critic, tau=self.soft_update_tau
        )

        return actor_loss.item(), critic_loss.item(), mu_v.mean().item()

    def soft_synchronize_models(self, source_model, target_model, tau):
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = tau * target_model_state[k] + (1.0 - tau) * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "ddpg_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.actor.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "ddpg_{0}_latest.pth".format(self.env_name)))

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)
        print("=============VALIDATION===============")
        for i in range(self.validation_num_episodes):
            episode_reward = 0

            self.test_env.reset()
            observation = self.test_env.get_init_observations()

            done = False

            while not done:
                action = self.actor.get_action(observation, exploration=False)

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        episode_reward_avg = np.average(episode_reward_lst)

        total_training_time = time.time() - self.total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

        print(
            "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                episode_reward_lst, episode_reward_avg, total_training_time
            )
        )
        return episode_reward_lst, episode_reward_avg

def main():
    # env = gym.make("Pendulum-v1")
    # test_env = gym.make("Pendulum-v1")
    card = HIL("qube_servo3_usb", "0")
    env = QuanserEnv(card)
    test_env = QuanserEnv(card)

    config = {
        "env_name": "quanser",                              # 환경의 이름
        "max_num_episodes": 200_000,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "steps_between_train": 32,                          # 훈련 사이의 환경 스텝 수
        "replay_buffer_size": 1_000_000,                    # 리플레이 버퍼 사이즈
        "learning_rate": 0.0003,                            # 학습율
        "gamma": 0.99,                                      # 감가율
        "soft_update_tau": 0.995,                           # DDPG Soft Update Tau
        "print_episode_interval": 1,                        # Episode 통계 출력에 관한 에피소드 간격
        "validation_time_steps_interval": 1_000,            # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,                       # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": -0.001,                # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
    }
    print(env.observation_space)
    print(env.action_space)
    use_wandb = True
    ddpg = DDPG(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    ddpg.train_loop()

if __name__ == "__main__":
    main()