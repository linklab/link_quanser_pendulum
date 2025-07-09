from quanser.hardware import MAX_STRING_LENGTH
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
        self.pwm_ch = array('I', [0])
        self.motor_enc_ch = array('I', [0])
        self.pend_enc_ch = array('I', [1])

        # LED channels
        self.led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

        # control params
        self.Ts = 0.02            # control timestep [s]
        self.max_steps = 1000    # after 1000 steps, truncate
        self.step_count = 0

        # for PID control
        self.Kp = 0.09
        self.Kd = 0.005

        # for estimate angular velocity
        self.prev_motor_angle = None
        self.prev_pend_angle = None

        # for last action observation
        self.last_action = 0.0 # radian

        # get initial motor count
        enc_val = array('l', [0])
        self.card.read_encoder(self.motor_enc_ch, 1, enc_val)
        self.init_count = enc_val[0]
        print(f"SET INIT COUNT: {self.init_count}")

        # initialize pendulum degree
        zero_ct = array('l', [0])
        self.card.set_encoder_counts(self.pend_enc_ch, len(self.pend_enc_ch), zero_ct)

        self.step_time = time.time()

    def observation_space(self):
        pass

    def action_space(self):
        pass

    def _get_motor_angle(self):
        enc_val = array('l', [0])
        self.card.read_encoder(self.motor_enc_ch, 1, enc_val)
        count = enc_val[0] - self.init_count
        return count * 2 * math.pi / 2048.0 # radian

    def _get_pendulum_angle(self):
        enc_val = array('l', [0])
        self.card.read_encoder(self.pend_enc_ch, 1, enc_val)
        raw_angle = enc_val[0] * 2 * math.pi / 2048.0
        angle = ((raw_angle + math.pi) % (2 * math.pi)) - math.pi
        return angle  # radian [-π, π], 6 o'clock: 0 radian

    def _get_motor_velocity(self, current):
        if self.prev_motor_angle is None:
            omega = 0.0
        else:
            omega = (current - self.prev_motor_angle) / self.Ts
        self.prev_motor_angle = current
        return omega

    def _get_pendulum_velocity(self, current):
        if self.prev_pend_angle is None:
            gamma = 0.0
        else:
            raw_diff = current - self.prev_pend_angle
            delta = ((raw_diff + math.pi) % (2 * math.pi)) - math.pi
            gamma = delta / self.Ts
        self.prev_pend_angle = current
        return gamma

    def get_init_observations(self):
        motor_angle = self._get_motor_angle()
        pendulum_angle = self._get_pendulum_angle()
        motor_angle_vel = self._get_motor_velocity(motor_angle)
        pendulum_angle_vel = self._get_pendulum_velocity(pendulum_angle)
        last_action = self.last_action
        return motor_angle, pendulum_angle, motor_angle_vel, pendulum_angle_vel, last_action # (rad, rad, rad/s, rad/s, rad)

    def reset(self):
        self.step_count = 0
        self.prev_motor_angle = None
        self.prev_pend_angle = None
        self.last_action = 0.0

        try:
            print("\n======RESET======")
            # # Reset LED blue
            # blue_led_values = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            # self.card.write_other(self.led_channels, len(self.led_channels), blue_led_values)

            # ③ PD 제어 파라미터
            duration = 20.0  # 제어 시간 (초) 10초 동안 reset할 시간 주어줌
            ang_tolerance = 10.0 # (degree) 절댓값 10도 이내로 reset시키면 즉시 reset 종료
            ang_vel_tolerance = 0.1 # (rad/s) 각속도 0.1 이내로 reset시키면 즉시 reset 종료
            steps = int(duration / self.Ts)

            init_val = array('l', [0])
            self.card.read_encoder(self.motor_enc_ch, 1, init_val)
            prev_error = (self.init_count - init_val[0]) * 2 * math.pi / 2048.0

            start = time.time()
            # ④ 제어 루프
            for i in range(steps):
                # 현재 각도 읽기
                enc_val = array('l', [0])
                self.card.read_encoder(self.motor_enc_ch, 1, enc_val)
                count = self.init_count - enc_val[0]  # target_angle - current_angle
                error = count * 2 * math.pi / 2048.0

                # 각속도 추정 (Finite difference)
                omega = (error - prev_error) / self.Ts
                prev_error = error

                # 허용 오차 이내로 reset시키면 즉시 reset 종료
                if abs(math.degrees(error)) < ang_tolerance and abs(omega) < ang_vel_tolerance:
                    print("======RESET COMPLETE======")
                    break

                # PD 제어
                duty = self.Kp * error + self.Kd * omega  # p_gain * angle + d_gain * angular_vel
                duty = max(min(duty, 0.3), -0.3)  # PWM 제한 [-0.1, +0.1]

                # 현재 모터 각도, 현재 모터 각속도, 현재 duty 값 출력
                if i % 5 == 0:
                    print(f"error = {math.degrees(error):+.2f} deg, omega = {omega:.1f} rad/s, duty = {duty:+.3f}")
                self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
                time.sleep(self.Ts)
            print(f"Reset time: {time.time() - start:.2f} sec")


        finally:
            # 모터 정지 및 앰프 OFF
            self.card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
            # reset 제대로 못하면?

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # print("step_time: ", time.time() - self.step_time)
        self.step_time = time.time()
        self.step_count += 1

        # # Step LED Red
        # red_led_values = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # self.card.write_other(self.led_channels, len(self.led_channels), red_led_values)
        # interpret action as target motor angle [rad]

        # =========================ACT=========================
        target_angle = float(actions.item())

        self.last_action = target_angle

        # read current motor angle
        current_angle = self._get_motor_angle()
        # estimate angular velocity
        omega = self._get_motor_velocity(current=current_angle)

        # PD position control to compute PWM duty
        error = target_angle - current_angle
        duty = self.Kp * error - self.Kd * omega
        duty = max(min(duty, 0.1), -0.1)
        self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))


        # wait for control period
        time.sleep(self.Ts)
        # ==================================================

        # read observations from hardware
        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        next_omega = self._get_motor_velocity(current=motor_angle)
        gamma = self._get_pendulum_velocity(current=pend_angle)
        next_obs = torch.tensor([motor_angle, pend_angle, next_omega, gamma, self.last_action], dtype=torch.float32)

        # estimate pendulum angular velocity

        raw_diff = pend_angle - math.pi
        theta = ((raw_diff + math.pi) % (2 * math.pi)) - math.pi

        # reward: -(theta^2 + 0.1 * gamma^2)
        reward_val = -(theta ** 2 + 0.1 * (gamma ** 2))
        reward = torch.tensor(reward_val, dtype=torch.float32)

        # # Success LED Green
        # if reward.item() > -0.01:
        #     green_led_values = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        #     self.card.write_other(self.led_channels, len(self.led_channels), green_led_values)

        # termination: motor angle exceeds ±130°
        terminated = abs(math.degrees(motor_angle)) > 130.0

        # truncation: max steps reached
        truncated = self.step_count >= self.max_steps

        # done flag
        dones = torch.tensor(terminated or truncated, dtype=torch.long)
        info = {}


        return next_obs, reward, dones, info
