import gymnasium as gym
import os
import torch
from actor_and_q_critic import QCritic
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class DqnTester:
    def __init__(self, env: gym.Env, qnet, env_name, current_dir):
        self.env = env

        self.model_dir = os.path.join(current_dir, "models")
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.video_dir = os.path.join(current_dir, "videos")
        if not os.path.exists(self.video_dir):
            os.mkdir(self.video_dir)

        self.env = gym.wrappers.RecordVideo(
            env=self.env, video_folder=self.video_dir,
            name_prefix="dqn_{0}_test_video".format(env_name)
        )

        self.qnet = qnet

        model_params = torch.load("/home/hyoseok/src/link_quanser_pendulum/models/dqn_cartpole-v1_500.0_qnet_20250826_221024.pth")
        self.qnet.load_state_dict(model_params)
        self.qnet.eval()

    def test(self):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = self.env.reset()
        time_steps = 0

        done = False

        while not done:
            time_steps += 1
            action = self.qnet.get_action_eps(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, _ = self.env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        self.env.close()
        print("[TOAL_STEPS: {0:3d}, EPISODE REWARD: {1:4.1f}".format(time_steps, episode_reward))



def main():
    ENV_NAME = "CartPole-v1"

    test_env = gym.make(ENV_NAME, render_mode="rgb_array")

    qnet = QCritic(n_features=4, n_actions=2)

    dqn_tester = DqnTester(
        env=test_env, qnet = qnet, env_name=ENV_NAME, current_dir=CURRENT_DIR
    )
    dqn_tester.test()

    test_env.close()

if __name__ == "__main__":
    main()