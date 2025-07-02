from quanser.hardware import MAX_STRING_LENGTH
from array   import array
import time
import math
import torch

from dummy_env import VecEnv
from quanser_control_pwm import reset

class QuanserEnv(VecEnv):
    def __init__(self, env, card):
        self.env = env # unwrapped env는 gymnasium "Pendulum-v1"
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

        # control params
        self.Ts = 0.1            # control timestep [s]
        self.max_steps = 1000    # after 1000 steps, truncate
        self.step_count = 0

        # for PID control
        self.Kp = 0.08
        self.Kd = 0.005
        self.prev_motor_angle = None

        # get initial motor count
        enc_val = array('l', [0])
        self.card.read_encoder(self.motor_enc_ch, 1, enc_val)
        self.init_count = enc_val[0]

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
        return enc_val[0] * 2 * math.pi / 2048.0 # radian

    def get_init_observations(self):
        motor_angle = self._get_motor_angle()
        pendulum_angle = self._get_pendulum_angle()
        return motor_angle, pendulum_angle # (rad, rad)

    def reset(self):
        reset(self.card)
        self.step_count = 0

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # interpret action as target motor angle [rad]
        target_angle = float(actions.item())

        # read current motor angle
        current_angle = self._get_motor_angle()
        # estimate angular velocity
        omega = 0.0
        if self.prev_motor_angle is not None:
            omega = (current_angle - self.prev_motor_angle) / self.Ts
        self.prev_motor_angle = current_angle

        # PD position control to compute PWM duty
        error = target_angle - current_angle
        duty = self.Kp * error - self.Kd * omega
        duty = max(min(duty, 0.1), -0.1)
        self.card.write_pwm(self.pwm_ch, 1, array('d', [duty]))

        # wait for control period
        time.sleep(self.Ts)

        # read observations from hardware
        motor_angle = self._get_motor_angle()
        pend_angle = self._get_pendulum_angle()
        next_obs = torch.tensor([motor_angle, pend_angle], dtype=torch.float32)

        # TODO
        reward = torch.tensor(0.0, dtype=torch.float32)

        # termination: motor angle exceeds ±130°
        terminated = abs(math.degrees(motor_angle)) > 130.0

        # truncation: max steps reached
        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        # done flag
        dones = torch.tensor(terminated or truncated, dtype=torch.long)
        info = {}


        return next_obs, reward, dones, info
