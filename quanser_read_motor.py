from quanser.hardware import HIL, HILError
from array   import array
import time
import math

def read_motor_angle(card):
    try:
        enc_ch  = array('I', [0])             # 모터 = 채널 0
        enc_val = array('i', [0])
        tach_ch_motor    = array('I', [14000])  # 14000 = 0번 엔코더의 velocity
        tach_val_motor   = array('d', [0.0])

        # ── (1) 첫 번째 읽기 → bias 저장 ───────────────────────────────
        card.read_encoder(enc_ch, 1, enc_val)
        bias_count = enc_val[0]               # 현재 카운트를 0° 기준으로 사용
        prev_t = time.perf_counter()
        # 영점 보정된 카운트를 라디안으로 바꾼 값
        prev_alpha = (bias_count - bias_count) * 2.0*math.pi / 2048.0          
        # ── (2) 계속 각도 출력 ────────────────────────────────────────
        while True:
            card.read_encoder(enc_ch, 1, enc_val)
            curr_t = time.perf_counter()
            count = enc_val[0] - bias_count   # 오프셋 보정

            alpha = count * 2*math.pi / 2048.0
            dt = curr_t - prev_t
            if dt > 0:
                omega = (alpha - prev_alpha) / dt
            else:
                omega = 0.0
            alpha_deg = alpha * 180/math.pi
            card.read_other(tach_ch_motor, 1, tach_val_motor)
            tach_vel_rad = tach_val_motor[0] * 2.0 * math.pi / 2048.0
            # 타코 속도와 엔코더 속도 비교
            print("motor_vel diff:", omega - tach_vel_rad)
            prev_alpha = alpha
            prev_t = curr_t
            # print("init count:", bias_count)
            print("motor count:", enc_val[0])
            # print(f"motor radian = {alpha:+.1f} rad")
            print(f"motor α = {alpha_deg:+.1f} °")
            time.sleep(0.004)
    except HILError as e:
        print(e.error_code)
        print(e.get_error_message())

    finally:
        pass

def main():
    try:
        card= HIL("qube_servo3_usb", "0")
    except HILError as e:
        print(e.error_code)
        print(e.get_error_message())

    finally:
        read_motor_angle(card=card)
        card.close()

if __name__ == "__main__":
    main()