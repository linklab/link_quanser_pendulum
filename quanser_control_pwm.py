from quanser.hardware import HIL, MAX_STRING_LENGTH
from array import array
import numpy as np, time, math

# -0.1, 0.1 duty 반복 제어 및 현재 모터 각도 출력
def test_pwm(card):
    try:
        # ① PWM 모드 켜기
        card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)        # 필수! :contentReference[oaicite:8]{index=8}
        input_channels = array('I', [1])
        output_channels = array('I', [0])
        num_input_channels = len(input_channels)
        num_output_channels = len(output_channels)
        card.set_digital_directions(input_channels, num_input_channels, output_channels, num_output_channels)

        enc_ch  = array('I', [0])             # 모터 = 채널 0
        enc_val = array('l', [0])
        # ── (1) 첫 번째 읽기 → bias 저장 ───────────────────────────────
        card.read_encoder(enc_ch, 1, enc_val)
        bias_count = enc_val[0]               # 현재 카운트를 0° 기준으로 사용
        print(f"zero-angle count: {bias_count} deg")
        # ② 앰프 Enable
        card.write_digital(array('I',[0]),1,array('I',[1]))

        pwm_ch  = array('I',[0])
        samples = 1.0
        Ts      = 0.1        # 1 kHz 루프
        start = time.time()
        for k in range(samples):
            card.read_encoder(enc_ch, 1, enc_val)
            count = enc_val[0] - bias_count   # 오프셋 보정

            alpha = count * 2*math.pi / 2048.0
            alpha_deg = alpha * 180/math.pi

            print(f"motor α = {alpha_deg:+.1f} °")
            if k % 2 == 0:
                duty = 0.15
            else:
                duty = -0.15
            card.write_pwm(pwm_ch, 1, array('d',[duty]))
            time.sleep(Ts)
        print(f"Elapsed time: {time.time() - start:.2f} sec")

    finally:
        card.write_pwm(array('I',[0]),1,array('d',[0.0]))
        card.write_digital(array('I',[0]),1,array('I',[0]))

def reset(card):
    try:
        # ① 모터 초기화 설정
        card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)
        input_channels = array('I', [1])
        output_channels = array('I', [0])
        card.set_digital_directions(input_channels, len(input_channels), output_channels, len(output_channels))
        card.write_digital(array('I',[0]),1,array('I',[1]))  # 앰프 ON

        # ② 채널 설정
        pwm_ch  = array('I', [0])
        enc_ch  = array('I', [0])
        enc_val = array('l', [0])

        # ③ PD 제어 파라미터
        Kp = 0.08     # stiffness (best gain: 0.09) +-5도 정도의 오차를 줄이는 효과
        Kd = 0.005     # damping (best gain: 0.005) 세게 반동 주었을 때, 펜듈럼으로 인한 외력으로 무한진동할 때가 있는데 댐핑 줄이니 해결
        Ts = 0.1   # 제어 주기 (초)
        duration = 500.0  # 제어 시간 (초)
        steps = int(duration / Ts)

        # ④ 초기 각도를 +0.0도로 설정
        card.read_encoder(enc_ch, 1, enc_val)
        target_count = enc_val[0]
        print("SET TARGET ANGLE")
        print(f"target_count: {target_count}")
        prev_alpha = target_count * 2 * math.pi / 2048.0

        start = time.time()
        # ④ 제어 루프
        for i in range(steps):
            # 현재 각도, 속도 읽기
            card.read_encoder(enc_ch, 1, enc_val)
            count = target_count - enc_val[0]  # target_angle - current_angle
            alpha = count * 2 * math.pi / 2048.0
            # 각속도 추정 (Finite difference)
            omega = (alpha - prev_alpha) / Ts
            prev_alpha = alpha

            # PD 제어
            duty = Kp * alpha + Kd * omega  # p_gain * angle + d_gain * angular_vel
            duty = max(min(duty, 0.1), -0.1)  # PWM 제한 [-0.1, +0.1]

            # 현재 모터 각도, 현재 모터 각속도, 현재 duty 값 출력
            if i % 5 == 0:
                print(f"alpha = {math.degrees(alpha):+.2f} deg, omega = {omega:.1f} rad/s, duty = {duty:+.3f}")
            card.write_pwm(pwm_ch, 1, array('d', [duty]))
            time.sleep(Ts)
        print(f"Elapsed time: {time.time() - start:.2f} sec")

    finally:
        # ⑤ 모터 정지 및 앰프 OFF
        card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
        card.write_digital(array('I', [0]), 1, array('I', [0]))


def main():
    card = HIL("qube_servo3_usb", "0")

    # test_pwm(card=card)  # -0.1, 0.1 duty 반복 제어 및 현재 모터 각도 출력
    reset(card=card)
    card.close()
if __name__ == "__main__":
    main()

