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
import torch.optim as optim
from actor_and_q_critic import DEVICE, MODEL_DIR, Actor, QCritic, ReplayBuffer, Transition

import wandb


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


class DDPG:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]
        custom_name = "hz_0_04_scale_0_3"
        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_") + custom_name

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
        self.train_start_time = None

        self.nosie_scale_scheduler = LinearNoiseScheduler(start_scale=1.0, end_scale=0.3, decay_steps=50_000)

    def train_loop(self) -> None:
        self.total_train_start_time = time.time()

        policy_loss = critic_loss = mu_v = 0.0

        is_terminated = False
        episode_reward_lst = np.zeros(shape=(10,), dtype=float)
        episode_reward_avg = 0.0  # average per 10 episodes
        episode_num = 0
        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0
            episode_steps = 0
            episode_num += 1
            observation = self.env.reset()
            done = False
            print("======EPISODE START====== ")
            while not done:
                episode_steps += 1
                self.train_start_time = time.time()
                self.time_steps += 1
                # action = torch.empty(1).uniform_(-1.0, 1.0)
                # action = torch.zeros(1)
                # if self.time_steps < self.training_start_steps:
                #     action = np.array(self.env.action_space.sample())
                #     action = np.clip(action, a_min=-1., a_max=1.)
                # else:
                #     scale = self.nosie_scale_scheduler.get_scale(self.time_steps)
                #     action = self.actor.get_action(observation, scale=scale)
                    # action = self.actor.get_action_e(self.env, observation, self.time_steps, self.epsilon_start, self.epsilon_end, self.epsilon_decay)
                scale = self.nosie_scale_scheduler.get_scale(self.time_steps)
                action = self.actor.get_action(observation, scale=scale)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                # if not self.best_saved and episode_reward_avg > self.episode_reward_avg_solved:
                #     self.best_count += 1
                #     if self.best_count >= 10: # 10번 설정한 목표에 도달하면 학습 종료
                #         print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                #         self.model_save(episode_reward_avg)
                #         is_terminated = True
                #         self.best_saved = True

            print("======EPISODE END====== ")
            episode_reward_lst[n_episode % 10] = episode_reward
            episode_reward_avg = np.average(episode_reward_lst)
            print("EPSILON: ", self.actor.epsilon)

            policy_loss_list = []
            critic_loss_list = []
            mu_v_list = []

            if self.time_steps > self.training_start_steps:
                training_num = np.min([100, episode_steps])
                training_num = np.max([training_num, 60])
                for _ in range(training_num):
                    policy_loss, critic_loss, mu_v = self.train()
                    policy_loss_list.append(policy_loss)
                    critic_loss_list.append(critic_loss)
                    mu_v_list.append(mu_v)

            if n_episode % 50 == 0 and n_episode > 100:
                from array import array
                green_led_values = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                card.write_other(led_channels, len(led_channels), green_led_values)
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

                if not self.best_saved and validation_episode_reward_mean > self.episode_reward_avg_solved:
                    self.best_count += 1
                    if self.best_count >= 10: # 10번 설정한 목표에 도달하면 학습 종료
                        print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                        self.model_save(validation_episode_reward_mean)
                        is_terminated = True
                        self.best_saved = True

            policy_loss_mean = np.mean(policy_loss_list)
            critic_loss_mean = np.mean(critic_loss_list)
            mu_v_mean = np.mean(mu_v_list)

            if n_episode % self.print_episode_interval == 0:
                print("####################################################################################")
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
    ) -> None:
        self.wandb.log(
            {
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Policy Loss": policy_loss,
                "[TRAIN] Critic Loss": critic_loss,
                "[TRAIN] mu_v": mu_v,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "[TRAIN] Noise Scale": scale,
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
        # with torch.no_grad():
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


def main():
    env = QuanserEnv(card)

    config = {
        "env_name": "quanser",                              # 환경의 이름
        "max_num_episodes": 200_000,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 128,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "replay_buffer_size": 50_000,                    # 리플레이 버퍼 사이즈
        "learning_rate": 0.0003,                            # 학습율
        "gamma": 0.999,                                      # 감가율
        "soft_update_tau": 0.995,                           # DDPG Soft Update Tau
        "print_episode_interval": 1,                        # Episode 통계 출력에 관한 에피소드 간격
        "episode_reward_avg_solved": 250.0,                   # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 50_000,
        "training_start_steps": 128,
        "seed": 42
    }
    random.seed(config["seed"])
    # 2) NumPy 시드 설정
    np.random.seed(config["seed"])
    # 3) PyTorch 시드 설정
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
        # CuDNN을 결정론적으로 동작하도록 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(env.observation_space)
    print(env.action_space)
    use_wandb = True
    ddpg = DDPG(env=env, test_env=env, config=config, use_wandb=use_wandb)
    ddpg.train_loop()

if __name__ == "__main__":
    # try:
    #     main(card)
    # except KeyboardInterrupt:
    #     print("\n###User end###")
    # finally:
    #     print("###HIL card disconnect###")
    #     try:
    #         card.close()
    #     except Exception as e:
    #         print(f"Error occured in card.close(): {e}")
    #     print("###program end###")
    #     sys.exit(0)
    main()