from quanser.hardware import MAX_STRING_LENGTH, HILError
from array   import array
import time
import math
import torch
import numpy as np
import gymnasium as gym

from dummy_env import VecEnv
from quanser_control_pwm import reset

class QuanserEnv(gym.Env):
    def __init__(self, card):
        # self.env = env # unwrapped env는 gymnasium "Pendulum-v1"
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

        # control params
        self.Ts = 0.01            # control timestep [s]
        self.max_steps = 1000    # after 1000 steps, truncate
        self.action_scale = 0.3

        # counter
        self.step_count = 0
        self.reset_count = 0

        # for PID control(0.8, 0.001)
        self.Kp = 0.8
        self.Kd = 0.02

        # for estimate angular velocity
        self.prev_motor_angle = None
        self.prev_pend_angle = None

        # for last action observation
        self.last_action = 0.0 # pwm duty cycle

        # initialize pendulum degree
        zero_ct = array('l', [0])
        self.card.set_encoder_counts(self.pend_enc_ch, len(self.pend_enc_ch), zero_ct)

        # get initial motor count
        self._reset_init_count()

        # self.step_time = time.time() # control timestep(self.Ts) + overhead time = 0.014~0.016

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

    def observation_space(self):
        return self.observation_space

    def action_space(self):
        return self.action_space

    def _get_motor_angle(self):
        enc_val = array('l', [0])
        self.motor_read0 = time.perf_counter()
        self.card.read_encoder(self.motor_enc_ch, 1, enc_val)
        count = enc_val[0] - self.init_count
        return count * 2 * math.pi / 2048.0 # radian

    def _get_pendulum_angle(self):
        enc_val = array('l', [0])
        self.pendulum_read0 = time.perf_counter()
        self.card.read_encoder(self.pend_enc_ch, 1, enc_val)
        raw_angle = enc_val[0] * 2 * math.pi / 2048.0
        angle = ((raw_angle + math.pi) % (2 * math.pi)) - math.pi
        return angle  # radian [-π, π], 6 o'clock: 0 radian

    def _get_motor_velocity(self, current):
        if self.prev_motor_angle is None:
            omega = 0.0
        else:
            angle_interval = time.perf_counter() - self.motor_read0
            omega = (current - self.prev_motor_angle) / angle_interval
        self.prev_motor_angle = current
        return omega

    def _get_pendulum_velocity(self, current):
        if self.prev_pend_angle is None:
            gamma = 0.0
        else:
            raw_diff = current - self.prev_pend_angle
            delta = ((raw_diff + math.pi) % (2 * math.pi)) - math.pi
            angle_interval = time.perf_counter() - self.pendulum_read0
            gamma = delta / angle_interval
        self.prev_pend_angle = current
        return gamma

    def _reset_init_count(self):
        push_max_count = 0
        push_min_count = 0
        push_max_cnt = 0
        push_min_cnt = 0

        while True:
            push_max_cnt += 1
            self.card.write_pwm(self.pwm_ch, 1, array('d', [0.08]))  # max push
            time.sleep(0.01)
            if push_max_cnt > 1000:
                enc_val = array('l', [0])
                self.card.read_encoder(self.motor_enc_ch, 1, enc_val)
                push_max_count = enc_val[0]
                break

        while True:
            push_min_cnt += 1
            self.card.write_pwm(self.pwm_ch, 1, array('d', [-0.08]))  # min push
            time.sleep(0.01)
            if push_min_cnt > 1000:
                enc_val = array('l', [0])
                self.card.read_encoder(self.motor_enc_ch, 1, enc_val)
                push_min_count = enc_val[0]
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
        motor_angle_vel_n = observation[2] / 10.0
        pendulum_angle_vel_n = observation[3] / 10.0
        last_action_n = observation[4] / self.action_scale
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
            # blue_led_values = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            # self.card.write_other(self.led_channels, len(self.led_channels), blue_led_values)

            push_max_cnt = 0
            while True:
                push_max_cnt += 1
                if self.reset_count % 2 == 0:
                    self.card.write_pwm(self.pwm_ch, 1, array('d', [-0.05]))
                else:
                    self.card.write_pwm(self.pwm_ch, 1, array('d', [0.05]))
                time.sleep(0.01)
                if push_max_cnt > 1000:
                    zero_ct = array('l', [0])
                    self.card.set_encoder_counts(self.pend_enc_ch, len(self.pend_enc_ch), zero_ct)
                    break

            init_val = array('l', [0])
            self.card.read_encoder(self.motor_enc_ch, 1, init_val)
            first_rad = init_val[0] * 2 * math.pi / 2048.0

            start = time.time()
            reset_counter = 0
            reset_success_num = 0

            if self.reset_count % 100 == 0:
                self._reset_init_count()

            read_enc_time = 0.0
            write_pwm_time = 0.0
            # ④ 제어 루프
            while True:
                c0 = time.perf_counter()
                reset_counter += 1
                time.sleep(self.Ts - (write_pwm_time + read_enc_time))
                enc_val = array('l', [0])
                motor_read0 = time.perf_counter()
                self.card.read_encoder(self.motor_enc_ch, 1, enc_val)
                motor_read1 = time.perf_counter()
                read_enc_time = motor_read1 - motor_read0

                init_rad = self.init_count * 2 * math.pi / 2048.0
                cur_rad = enc_val[0] * 2 * math.pi / 2048.0
                error_rad = init_rad - cur_rad

                if abs(error_rad) < 0.1:
                    reset_success_num += 1
                else:
                    reset_success_num = 0

                if reset_success_num > 100:
                    break

                angle_interval = time.perf_counter() - motor_read0
                if reset_counter == 1:
                    omega = (cur_rad - first_rad) / self.Ts
                else:
                    omega = (cur_rad - prev_rad) / self.Ts
                prev_rad = cur_rad

                # PD 제어
                duty = self.Kp * error_rad - self.Kd * omega  # p_gain * angle + d_gain * angular_vel
                duty = max(min(duty, 0.04), -0.04)

                # 현재 모터 각도, 현재 모터 각속도, 현재 duty 값 출력
                pwm_write0 = time.perf_counter()
                self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
                pwm_write1 = time.perf_counter()
                write_pwm_time = pwm_write1 - pwm_write0
                c1 = time.perf_counter()
                # print(f"reset_control_time: {(c1 - c0):.4f}")
            print(f"Reset time: {time.time() - start:.2f} sec")

        finally:
            # 모터 정지 및 앰프 OFF
            self.card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
            print("\n======RESET END======")
            obs = self.get_init_observations()
            obs = self.noramlize_observation(obs)
            return obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        # while (time.time() - self.step_time) < 0.02:
        #     time.sleep(0.00001)

        # print("step_time: ", time.time() - self.step_time)
        # self.step_time = time.time()
        self.step_count += 1


        # # Step LED Red
        # red_led_values = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # self.card.write_other(self.led_channels, len(self.led_channels), red_led_values)
        # interpret action as target motor angle [rad]

        # =========================ACT=========================
        try:
            c0 = time.perf_counter()
            pwm = float(actions.item()) * self.action_scale
            self.last_action = pwm
            pwm_buf = array('d', [pwm])
            pwm_write0 = time.perf_counter()
            self.card.write_pwm(self.pwm_ch, 1, pwm_buf)
            pwm_write1 = time.perf_counter()
            write_pwm_time = pwm_write1 - pwm_write0

            # wait for control period
            # ==================================================

            # read observations from hardware
            read_enc0 = time.perf_counter()
            motor_angle = self._get_motor_angle()
            pend_angle = self._get_pendulum_angle()
            read_enc1 = time.perf_counter()
            read_enc_time = read_enc1 - read_enc0
            time.sleep(self.Ts - (write_pwm_time + read_enc_time))
            next_omega = self._get_motor_velocity(current=motor_angle)
            gamma = self._get_pendulum_velocity(current=pend_angle)
            next_obs = torch.tensor([motor_angle, pend_angle, next_omega, gamma, self.last_action], dtype=torch.float32)
            next_obs = self.noramlize_observation(next_obs)

            # print("\nn_motor_angle: ", next_obs[0].item())
            # print("n_pendulum_angle: ", next_obs[1].item())
            # print("n_motor_vel: ", next_obs[2].item())
            # print("n_pendulum_vel: ", next_obs[3].item())
            # print("n_last_action: ", next_obs[4].item())
            # estimate pendulum angular velocity

            raw_diff = pend_angle
            theta = ((raw_diff + math.pi) % (2 * math.pi)) - math.pi

            reward = abs(theta)

            # reward: -(theta^2 + 0.1 * gamma^2 + 0.001 * pwm_duty^2)
            # reward = -(theta ** 2 + 0.1 * (gamma ** 2) + 0.001 * (pwm ** 2))
            # reward = torch.tensor(reward_val, dtype=torch.float32)
            # # Success LED Green
            # if reward.item() > -0.01:
            #     green_led_values = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            #     self.card.write_other(self.led_channels, len(self.led_channels), green_led_values)

            # termination: motor angle exceeds ±130°
            terminated = abs(math.degrees(motor_angle)) > 100.0

            # truncation: max steps reached
            truncated = self.step_count >= self.max_steps

            info = {}

            c1 = time.perf_counter()
            # print(f"step_control_time: {(c1 - c0):.4f}")
            # if terminated or truncated:
            #     self.card.write_pwm(self.pwm_ch, 1, array('d', [0.0]))
            #     time.sleep(1)
            return next_obs, reward, terminated, truncated, info
        except HILError as e:
            print(e.error_code)
            print(e.get_error_message())
        finally:
            pass
