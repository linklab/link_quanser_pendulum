from quanser.hardware import MAX_STRING_LENGTH, HILError
from array   import array
import time
import math
import torch
import numpy as np
import gymnasium as gym
import copy

from dummy_env import VecEnv
from quanser_control_pwm import reset

class QuanserEnv(gym.Env):
    def __init__(self, card):
        self.card = card

        # PWM 모드 켜기
        self.card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)        # 필수! :contentReference[oaicite:8]{index=8}
        input_channels = array('I', [1])
        output_channels = array('I', [0])
        num_input_channels = len(input_channels)
        num_output_channels = len(output_channels)
        self.card.set_digital_directions(input_channels, num_input_channels, output_channels, num_output_channels)
        self.card.write_digital(array('I',[0]),1,array('I',[1]))

        # channels
        self.pwm_ch = array('I', [0]) # pwm output channel
        self.motor_enc_ch = array('I', [0]) # encoder input channel
        self.pend_enc_ch = array('I', [1]) # encoder input channel

        # LED channels
        self.led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

        # encoder value buffer
        self.motor_enc_val = array('l', [0])
        self.pend_enc_val = array('l', [0])

        # control params
        self.Ts = 0.01            # control timestep [s]
        self.max_steps = 300    # after 1000 steps, truncate
        self.action_scale = 0.3

        # counter
        self.step_count = 0
        self.reset_count = 0
        self.gamma_over_count = 0

        # for PID control
        self.Kp = 0.8
        self.Kd = 0.02

        # for estimate angular velocity
        self.prev_motor_angle = None
        self.prev_pend_angle = None

        # for last action observation
        self.last_action = 0.0 # pwm duty cycle

        # get initial motor count
        self._reset_init_count()

        low_obs = np.array([
            -2.5,       # motor_angle(radian)
            -math.pi,   # pendulum_angle(radian)
            -np.inf,    # motor_ang_vel(radian/sec)
            -np.inf,    # pendulum_ang_vel(radian/sec)
            -1.0        # last_action(pwm duty cycle)
        ], dtype=np.float32)

        high_obs = np.array([
             2.5,
             math.pi,
             np.inf,
             np.inf,
             1.0
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low_obs,
                                                high=high_obs,
                                                dtype=np.float32)

        self.action_space = gym.spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                           high=np.array([1.0], dtype=np.float32),
                                           dtype=np.float32)
        
        self.pend_init_count = 0

    def observation_space(self):
        return self.observation_space

    def action_space(self):
        return self.action_space

    def _get_motor_angle(self):
        if self.prev_motor_angle is None:
            self.prev_motor_read = None
        else:
            self.prev_motor_read = copy.deepcopy(self.motor_read0)
        self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
        self.motor_read0 = time.perf_counter()
        count = self.motor_enc_val[0] - self.init_count
        return count * 2 * math.pi / 2048.0 # radian

    def _get_pendulum_angle(self):
        if self.prev_pend_angle is None:
            self.prev_pend_read = None
        else:
            self.prev_pend_read = copy.deepcopy(self.pendulum_read0)
        self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
        self.pendulum_read0 = time.perf_counter()
        raw_angle = (self.pend_enc_val[0] - self.pend_init_count) * 2 * math.pi / 2048.0
        angle = ((raw_angle + math.pi) % (2 * math.pi)) - math.pi
        return angle  # radian [-π, π], 6 o'clock: 0 radian

    def _get_motor_velocity(self, current):
        if self.prev_motor_angle is None:
            omega = 0.0
        else:
            angle_interval = self.motor_read0 - self.prev_motor_read
            omega = (current - self.prev_motor_angle) / angle_interval
        self.prev_motor_angle = current # radian
        return omega

    def _get_pendulum_velocity(self, current):
        if self.prev_pend_angle is None:
            gamma = 0.0
        else:
            raw_diff = (self.pend_enc_val[0] - self.prev_pend_angle) * 2 * math.pi / 2048.0
            angle_interval = self.pendulum_read0 - self.prev_pend_read
            gamma = raw_diff / angle_interval
        self.prev_pend_angle = copy.deepcopy(self.pend_enc_val[0]) # encoder count -> 12 o'clock: overflow gamma problem
        return gamma

    def _reset_init_count(self):
        push_max_count = 0
        push_min_count = 0
        push_max_cnt = 0
        push_min_cnt = 0

        while True:
            push_max_cnt += 1
            self.card.write_pwm(self.pwm_ch, 1, array('d', [-0.06]))  # max push
            time.sleep(0.01)
            if push_max_cnt > 300:
                self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
                push_max_count = self.motor_enc_val[0]
                break

        while True:
            push_min_cnt += 1
            self.card.write_pwm(self.pwm_ch, 1, array('d', [0.06]))  # min push
            time.sleep(0.01)
            if push_min_cnt > 300:
                self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
                push_min_count = self.motor_enc_val[0]
                break

        self.init_count = int((push_max_count + push_min_count) / 2.0)
        print("set init count: ", self.init_count)

    def get_init_observations(self):
        motor_angle = self._get_motor_angle()
        pendulum_angle = self._get_pendulum_angle()
        motor_angle_vel = self._get_motor_velocity(motor_angle)
        pendulum_angle_vel = self._get_pendulum_velocity(pendulum_angle)
        last_action = self.last_action
        observation_t = torch.tensor([motor_angle, pendulum_angle, motor_angle_vel, pendulum_angle_vel, last_action], dtype=torch.float32)
        return observation_t

    def noramlize_observation(self, observation):
        motor_angle_n = observation[0] / 1.8
        pendulum_angle_n = observation[1] / math.pi
        motor_angle_vel_n = observation[2] / 20.0
        pendulum_angle_vel_n = observation[3] / 20.0
        last_action_n = observation[4]
        observation_n = torch.tensor([motor_angle_n, pendulum_angle_n, motor_angle_vel_n, pendulum_angle_vel_n, last_action_n], dtype=torch.float32)
        return observation_n

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        self.prev_motor_angle = None
        self.prev_pend_angle = None
        self.last_action = 0.0

        try:
            print("\n======RESET START======")
            # # Reset LED blue
            blue_led_values = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            self.card.write_other(self.led_channels, len(self.led_channels), blue_led_values)

            start = time.time()
            reset_counter = 0
            reset_success_num = 0
            omega = 0.0

            if self.reset_count % 50 == 0:
                self._reset_init_count()

            pen_init_count_list = []

            while True:
                reset_counter += 1
                cur_rad = self._get_motor_angle()
                error_rad = -cur_rad

                if abs(error_rad) < 0.1 and abs(omega) < 0.31:
                    reset_success_num += 1
                else:
                    reset_success_num = 0
                    pen_init_count_list = []

                if reset_success_num > 50:
                    self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
                    pen_init_count_list.append(self.pend_enc_val[0])
                    if len(pen_init_count_list) > 50:
                        pend_init_count_mean = int(np.mean(pen_init_count_list))
                        print("pendulum init count diff: ", (self.pend_init_count - pend_init_count_mean) % 2048)
                        self.pend_init_count = pend_init_count_mean
                        break

                if reset_counter == 1:
                    omega = 0.0
                else:
                    omega = (cur_rad - prev_rad) / self.Ts
                prev_rad = cur_rad

                # PD 제어
                duty = self.Kp * 2 * error_rad - self.Kd * 0.5 * omega
                duty = max(min(duty, 0.03), -0.03)  # PWM 제한 [-0.03, +0.03]

                self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
                time.sleep(0.01)

                if reset_counter > 500 and reset_success_num == 0:
                    self.card.write_pwm(self.pwm_ch, 1, array('d', [0.06]))
                    time.sleep(0.01)
                    reset_counter = 0
                    print("FAILED TO RESET, RETRYING... error_rad: ", abs(error_rad), " ,abs(omega): ", abs(omega))

            print(f"Reset time: {time.time() - start:.2f} sec")

        finally:
            # 모터 정지 및 앰프 OFF
            self.card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
            print("\n======RESET END======")
            obs = self.get_init_observations()
            obs = self.noramlize_observation(obs)
            self.step_time = time.perf_counter()
            return obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:

        self.step_count += 1
        terminated = False

        # # Step LED Red
        red_led_values = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.card.write_other(self.led_channels, len(self.led_channels), red_led_values)
        # interpret action as target motor angle [rad]

        # =========================ACT=========================
        pwm = float(actions.item()) * self.action_scale
        pwm_buf = array('d', [pwm])
        self.card.write_pwm(self.pwm_ch, 1, pwm_buf)
        self.step_time = time.perf_counter()
        last_pwm = copy.deepcopy(self.last_action)
        self.last_action = actions.item()

        # wait for control period
        while (time.perf_counter() - self.step_time) < 0.02:
            time.sleep(0.001)
        # =====================================================

        # read observations from hardware
        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        next_omega = self._get_motor_velocity(current=motor_angle)
        gamma = self._get_pendulum_velocity(current=pend_angle)
        next_obs = torch.tensor([motor_angle, pend_angle, next_omega, gamma, last_pwm], dtype=torch.float32)
        next_obs = self.noramlize_observation(next_obs)

        # print("\nn_motor_angle: ", next_obs[0].item())
        # print("n_pendulum_angle: ", next_obs[1].item())
        # print("n_motor_vel: ", next_obs[2].item())
        # print("n_pendulum_vel: ", next_obs[3].item())
        # print("n_last_action: ", next_obs[4].item())

        raw_diff = pend_angle
        # if abs(raw_diff) < 0.1:
        #     plus_rew = raw_diff
        # else:
        #     plus_rew = 0.0
        # reward = abs(raw_diff) + plus_rew ** 2
        reward = abs(raw_diff)**2
        reward /= (math.pi)**2

        # termination: motor angle exceeds ±90°
        if abs(math.degrees(motor_angle)) > 80.0:
            terminated = True

        if abs(gamma) > 30:
            self.gamma_over_count += 1
            print("GAMMA OVER: ", gamma)
            terminated = True
            if self.gamma_over_count == 1:
                initailize_pen = True

        # if abs(gamma) > 80:
        #     print(gamma, "!!!!!!!!!!!")
        #     prev_pend_rad = pend_angle - gamma * (self.pendulum_read0 - self.prev_pend_read)
        #     print("prev pen rad: ", prev_pend_rad, ", cur pen rad: ", pend_angle)
        #     print("pen diff time: ", self.pendulum_read0 - self.prev_pend_read)
        #     initailize_pen = True

        # truncation: max steps reached
        truncated = self.step_count >= self.max_steps

        info = {}

        if terminated or truncated:
            blue_led_values = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            self.card.write_other(self.led_channels, len(self.led_channels), blue_led_values)
            reset_counter = 0
            target_angle = 0.0
            reset_success_num = 0
            omega = 0.0
            while True:
                reset_counter += 1
                self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
                cur_rad = (self.motor_enc_val[0] - self.init_count) * 2 * math.pi / 2048.0
                error_rad = target_angle - cur_rad

                if abs(error_rad) < 0.1:
                    reset_success_num += 1
                else:
                    reset_success_num = 0

                if reset_success_num > 50:
                    self.card.write_pwm(self.pwm_ch, 1, array('d', [0.0]))
                    break

                if reset_counter == 1:
                    omega = next_omega
                else:
                    omega = (cur_rad - prev_rad) / self.Ts
                prev_rad = cur_rad

                duty = self.Kp * 2 * error_rad - self.Kd * 0.5 * omega
                # PD 제어
                if reset_counter < 4:
                    duty = max(min(duty, 0.2), -0.2)
                else:
                    duty = max(min(duty, 0.03), -0.03)

                self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
                time.sleep(0.01)

        return next_obs, reward, terminated, truncated, info