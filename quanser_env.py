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

from collections import deque
np.set_printoptions(precision=5, suppress=True)


class QuanserEnv(gym.Env):
    def __init__(self, card):
        self.card = card

        # PWM mode setup
        self.card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)        # 필수! :contentReference[oaicite:8]{index=8}
        input_channels = array('I', [1])
        output_channels = array('I', [0])
        num_input_channels = len(input_channels)
        num_output_channels = len(output_channels)
        self.card.set_digital_directions(input_channels, num_input_channels, output_channels, num_output_channels)
        self.card.write_digital(array('I',[0]),1,array('I',[1]))

        # channels
        self.pwm_ch = array('I', [0]) # pwm output channel
        self.motor_enc_ch = array('I', [0]) # encoder input channel (motor angle)
        self.pend_enc_ch = array('I', [1]) # encoder input channel (pendulum angle)
        self.tach_ch = array('I', [14001]) # tachometer input channel (pendulum angular velocity)
        self.tach_motor_ch = array('I', [14000]) # tachometer input channel (motor angular velocity)

        # LED channels
        self.led_channels = np.array([11000, 11001, 11002], dtype=np.uint32)

        # encoder value buffer
        self.motor_enc_val = array('i', [0])  # motor angle
        self.pend_enc_val = array('i', [0])  # pendulum angle
        self.tach_vel_val = array('d', [0.0])  # pendulum angular velocity
        self.tach_motor_vel_val = array('d', [0.0])  # motor angular velocity

        # control params
        self.Ts = 0.006            # cntrol timestep [s]
        self.max_steps = 2000    # after 1000 steps, truncate
        self.action_scale = 0.35  # scale for action (pwm duty cycle)

        # total time steps
        self.time_steps = 0
        # counter
        self.step_count = 0
        self.reset_count = 0
        self.omega_over_count = 0

        # for PID control
        self.Kp = 0.8
        self.Kd = 0.02

        # for estimate angular velocity
        self.prev_motor_angle = None
        self.prev_pend_angle = None

        # for last action observation
        self.last_action = 0.0 # pwm duty cycle

        self.std_error = deque(maxlen=2000)

        # for initialization pendulum count
        self.pen_init_count_list = []

        low_obs = np.array([
            -2.5,       # motor_angle(radian)
            -math.pi,   # pendulum_angle(radian)
            -np.inf,    # motor_ang_vel(radian/sec)
            -np.inf,    # pendulum_ang_vel(radian/sec)
            -5.0        # pendulum_spin_num(integer)
        ], dtype=np.float32)

        high_obs = np.array([
             2.5,
             math.pi,
             np.inf,
             np.inf,
             5.0
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low_obs,
                                                high=high_obs,
                                                dtype=np.float32)

        self.action_space = gym.spaces.Box(low=np.array([-1.0], dtype=np.float32),
                                           high=np.array([1.0], dtype=np.float32),
                                           dtype=np.float32)
        
        # zero encoder counts
        self.init_count = 0
        self.pend_init_count = 0

        # get initial motor count
        self._reset_init_count()

        self.step_time = None

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

    def _get_motor_velocity(self):
        self.card.read_other(self.tach_motor_ch, 1, self.tach_motor_vel_val)
        tach_motor_vel_rad = self.tach_motor_vel_val[0] * 2.0 * math.pi / 2048.0
        return tach_motor_vel_rad

    def _get_pendulum_velocity(self):
        self.card.read_other(self.tach_ch, 1, self.tach_vel_val)
        tach_vel_rad = self.tach_vel_val[0] * 2.0 * math.pi / 2048.0
        return tach_vel_rad
    
    def _get_motor_acceleration(self, cur_motor_vel):
        if self.prev_motor_vel is None:
            self.prev_motor_vel = cur_motor_vel
            return 0.0
        else:
            acc = (cur_motor_vel - self.prev_motor_vel)
            self.prev_motor_vel = cur_motor_vel
            return acc
        
    def _get_pend_acceleration(self, cur_pend_vel):
        if self.prev_pend_vel is None:
            self.prev_pend_vel = cur_pend_vel
            return 0.0
        else:
            acc = (cur_pend_vel - self.prev_pend_vel)
            self.prev_pend_vel = cur_pend_vel
            return acc

    def _reset_init_count(self):
        print("calibrate motor init count...")
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

    def _reset_pendulum_init_count(self) -> None:
        self.card.read_encoder(self.pend_enc_ch, 1, self.pend_enc_val)
        self.pen_init_count_list.append(copy.deepcopy(self.pend_enc_val[0]))
        if len(self.pen_init_count_list) > 100:
            pend_init_count_mean = int(np.mean(self.pen_init_count_list))
            print("pendulum init count diff: ", (self.pend_init_count - pend_init_count_mean)%2048)
            self.pend_init_count = pend_init_count_mean

    def get_init_observations(self):
        motor_angle = self._get_motor_angle()
        pendulum_angle = self._get_pendulum_angle()
        motor_angle_vel = self._get_motor_velocity()
        pendulum_angle_vel = self._get_pendulum_velocity()
        # motor_acc = self._get_motor_acceleration(motor_angle_vel)
        # pend_acc = self._get_pend_acceleration(pendulum_angle_vel)
        pend_spin_num = (self.pend_init_count - self.pend_enc_val[0]) / 2048.0
        # observation_t = torch.tensor([motor_angle, pendulum_angle, motor_angle_vel, pendulum_angle_vel, pend_spin_num], dtype=torch.float32)
        observation_t = torch.tensor([motor_angle, math.sin(pendulum_angle), math.cos(pendulum_angle), motor_angle_vel, pendulum_angle_vel, pend_spin_num], dtype=torch.float32)
        return observation_t

    def noramlize_observation(self, observation):
        motor_angle_n = observation[0] / 1.8
        motor_angle_vel_n = observation[3] / 20.0
        pendulum_angle_vel_n = observation[4] / 40.0
        # motor_acc_n = observation[4] / 10.0
        # pendulum_acc_n = observation[5] / 20.0
        pendulum_spin_num = observation[5] / 5.0
        observation_n = torch.tensor([motor_angle_n, observation[1], observation[2], motor_angle_vel_n, pendulum_angle_vel_n, pendulum_spin_num], dtype=torch.float32)
        return observation_n

    def reset(self):
        self.step_count = 0
        self.reset_count += 1
        self.prev_motor_angle = None
        self.prev_pend_angle = None
        self.prev_motor_vel = None
        self.prev_pend_vel = None
        self.step_time = None
        self.last_action = 0.0
        self.omega_over_count = 0
        self.std_error = deque(maxlen=2000)

        print("\n======RESET START======")
        # Reset LED blue
        blue_led_values = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.card.write_other(self.led_channels, len(self.led_channels), blue_led_values)

        start = time.time()
        reset_counter = 0
        reset_success_num = 0
        omega = 0.0

        if self.reset_count % 5 == 0:
            self._reset_init_count()

        thresh_hold_pen_vel = 0.1
        thresh_hold_error_rad = 0.1
        reset_done = False

        while True:
            reset_counter += 1
            cur_rad = self._get_motor_angle()
            error_rad = -cur_rad
            pen_vel = self._get_pendulum_velocity()
            # print("error_rad:",error_rad)

            if abs(error_rad) < thresh_hold_error_rad and abs(pen_vel) < thresh_hold_pen_vel:
                reset_success_num += 1
            else:
                reset_success_num = 0
                self.pen_init_count_list = []

            if reset_success_num > 50:
                reset_done = True

            omega = self._get_motor_velocity()

            # PD control
            duty = self.Kp * 2 * error_rad - self.Kd * omega
            duty = max(min(duty, 0.03), -0.03)

            self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
            time.sleep(0.005)

            if reset_counter > 1000 and reset_success_num == 0:
                thresh_hold_pen_vel += 0.05
                thresh_hold_error_rad += 0.05
                reset_counter = 0
                if abs(error_rad) > 0.3:
                    self._reset_init_count()
                    print("RESET FAILED, RE-CALIBRATE MOTOR INIT COUNT")
                else:
                    print("FAILED TO RESET, RETRYING... error_rad: ", abs(error_rad), " ,abs(pend_val): ", abs(pen_vel))
            
            if reset_done:
                self.card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
                break
        
        self.pen_init_count_list = []
        for _ in range(101):
            self._reset_pendulum_init_count()
            time.sleep(0.003)

        print("\n======RESET END======")
        print(f"Reset time: {time.time() - start:.2f} sec")
        obs = self.get_init_observations()
        obs = self.noramlize_observation(obs)
        self.step_time = time.perf_counter()
        # Step LED Red
        red_led_values = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.card.write_other(self.led_channels, len(self.led_channels), red_led_values)
        self.prev_sign = None
        self.contiue_sign = 0
        self.step_time = time.perf_counter()
        return obs

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        self.time_steps += 1
        self.step_count += 1
        terminated = False

        # =========================ACT=========================
        # pwm = float(actions.item()) * self.action_scale
        # pwm_buf = array('d', [pwm])
        # self.card.write_pwm(self.pwm_ch, 1, pwm_buf)
        last_pwm = copy.deepcopy(self.last_action)
        self.last_action = actions.item()

        # while (time.perf_counter() - self.step_time) > self.Ts:
        #     time.sleep(0.0001)

        # print(pwm, end=' ')

        # current_sign = 0
        # if pwm > 0:
        #     current_sign = 1
        # elif pwm < 0:
        #     current_sign = -1

        # if self.prev_sign is not None and current_sign != self.prev_sign and current_sign != 0:
        #     print(f"부호 변경! (이전 부호: {self.prev_sign}, 현재 부호: {current_sign}), contiue_sign: {self.contiue_sign}")
        #     self.contiue_sign = 0
        # else:
        #     self.contiue_sign += 1

        # self.prev_sign = current_sign

        # read observations from hardware
        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        next_omega = self._get_motor_velocity()
        gamma = self._get_pendulum_velocity()
        # motor_acc = self._get_motor_acceleration(next_omega)
        # pend_acc = self._get_pend_acceleration(gamma)
        pend_spin_num = (self.pend_init_count - self.pend_enc_val[0]) / 2048.0
        # next_obs = torch.tensor([motor_angle, pend_angle, next_omega, gamma, pend_spin_num], dtype=torch.float32)
        next_obs = torch.tensor([motor_angle, math.sin(pend_angle), math.cos(pend_angle), next_omega, gamma, pend_spin_num], dtype=torch.float32)
        next_obs = self.noramlize_observation(next_obs)

        # compute reward
        reward = abs(pend_angle)**2
        reward /= math.pi**2

        if abs(pend_angle) > 2.96706: # abs(170deg)
            reward *= 2

        reward += 0.1

        # motor_angle_n = motor_angle / 1.8
        # motor_vel_n   = next_omega / 20.0
        
        # increase penalty
        # lambda_pos = self.motor_penalty_scheduler(0.05, 0.1, 150_000)
        # lambda_vel = self.motor_penalty_scheduler(0.01, 0.05, 150_000)

        # penalty = lambda_pos * (motor_angle_n**2) + lambda_vel * (motor_vel_n**2)

        # reward -= penalty

        # termination conditions
        if abs(pend_spin_num) > 5.0:
            print("PENDULUM SPIN OVER: ", pend_spin_num)
            terminated = True
            # self._reset_pendulum_init_count()  # calibrate pendulum init count
            # print("calibrate pendulum init count...")

        if abs(math.degrees(motor_angle)) > 90.0:
            print("MOTOR ANGLE OVER: ", math.degrees(motor_angle))
            terminated = True
            # self._reset_init_count()  # calibrate motor init count

        if abs(gamma) > 40.0:
            print("PENDULUM VELOCITY OVER: ", gamma)
            terminated = True
            # self._reset_pendulum_init_count()  # calibrate pendulum init count
            # print("calibrate pendulum init count...")

        # truncation: max steps reached
        truncated = self.step_count >= self.max_steps
        if truncated:
            print("TRUNCATED")
        info = {}

        if terminated or truncated:
            blue_led_values = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            self.card.write_other(self.led_channels, len(self.led_channels), blue_led_values)
            self.reset_helper()

        return next_obs, reward, terminated, truncated, info
    
    def reset_helper(self):
        reset_counter = 0
        target_angle = 0.0
        reset_success_num = 0
        omega = 0.0
        while True:
            reset_counter += 1
            # prevent infinite loop
            if reset_counter > 3000:
                break
            self.card.read_encoder(self.motor_enc_ch, 1, self.motor_enc_val)
            cur_rad = (self.motor_enc_val[0] - self.init_count) * 2 * math.pi / 2048.0
            error_rad = target_angle - cur_rad

            if abs(error_rad) < 0.2:
                reset_success_num += 1
            else:
                reset_success_num = 0

            if reset_success_num > 150:
                self.card.write_pwm(self.pwm_ch, 1, array('d', [0.0]))
                break

            omega = self._get_motor_velocity()

            duty = self.Kp * error_rad - self.Kd * omega

            # PD control
            if cur_rad > 1.57:
                duty = -0.4
            elif cur_rad < -1.57:
                duty = 0.4
            else:
                duty = max(min(duty, 0.2), -0.2)

            self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))
            time.sleep(0.005)
    
    def motor_penalty_scheduler(self, start, end, frac):
        t = min(1.0, max(0.0, self.time_steps / frac))
        return start + (end - start) * t
    
    def apply_action(self, actions):
        pwm = float(actions.item()) * self.action_scale
        pwm_buf = array('d', [pwm])
        self.card.write_pwm(self.pwm_ch, 1, pwm_buf)
