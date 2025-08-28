# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os

import gymnasium as gym
import torch

from quanser.hardware import HIL

from actor_and_q_critic import MODEL_DIR, Actor
from quanser_env import QuanserEnv
import time

card = HIL("qube_servo3_usb", "0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(env: QuanserEnv, actor: Actor, num_episodes: int) -> None:
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = actor.get_action(observation, exploration=False)
            env.apply_action(action)
            time.sleep(0.006)

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(i, episode_steps, episode_reward))


def main_play(num_episodes: int, env_name: str, sub_model_dir: str) -> None:
    env = QuanserEnv(card)

    actor = Actor(n_features=6, n_actions=1)
    # model_params = torch.load(os.path.join(MODEL_DIR, sub_model_dir, "td3_quanser_3961.0_actor_20250815_004857.pth"))
    model_params = torch.load("/home/link/src/link_quanser_pendulum/models/td3_quanser_6ms_0.35scale/td3_quanser_3971.4_actor_20250808_191231.pth", map_location=device)
    actor.load_state_dict(model_params)
    actor.to(device)
    actor.eval()

    test(env, actor, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 30
    ENV_NAME = "quanser_actor"
    SUB_MODEL_DIR = "best"

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME, sub_model_dir = SUB_MODEL_DIR)