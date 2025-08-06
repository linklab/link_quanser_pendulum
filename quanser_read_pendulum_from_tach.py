#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qube-Servo 3  │  엔코더 θ(t)  +  타코 ω(t) 혼합 보정 예제
"""


import math, time
from array import array
from quanser.hardware import HIL, HILError


# ----------------- 하드웨어 매핑 -----------------
MOTOR_CH       = array('I', [0])
PEND_CH        = array('I', [1])        # 펜듈럼 엔코더
TACHO_CH       = array('I', [14001])    # pendulum tacho velocity [rad/s]

CNT_PER_REV    = 2048               # 8192 edge/rev
DT             = 0.002                  # 500 Hz 필터 주기

# ----------------- 필터 파라미터 -----------------
ω_LOW          =  3.0     #  rad/s ↓ ⇒ 전적으로 엔코더 θ
ω_HIGH         = 15.0     #  rad/s ↑ ⇒ 전적으로 타코 ∫ωdt
# -------------------------------------------------


def counts_to_rad(count: int) -> float:
    return count * (2.0 * math.pi / CNT_PER_REV)


def complementary_filter(angle_enc, angle_int, omega):
    """속도 크기(ω)에 따라 두 각도를 가중합해서 반환"""
    abs_w = abs(omega)

    if abs_w <= ω_LOW:
        alpha = 0.0                # 0 → 전부 encoder
    elif abs_w >= ω_HIGH:
        alpha = 1.0                # 1 → 전부 tacho 적분
    else:
        # 선형 보간
        alpha = (abs_w - ω_LOW) / (ω_HIGH - ω_LOW)

    # θ_hat = (1-α)·θ_enc + α·θ_tacho_int
    return (1.0 - alpha) * angle_enc + alpha * angle_int, alpha


def main():
    card = HIL("qube_servo3_usb", "0")
    try:
        # ---------- 초기화 ----------
        # 원하는 ‘0 rad’ 위치에서 엔코더 영점
        zero_ct = array('l', [0])
        card.set_encoder_counts(PEND_CH, len(PEND_CH), zero_ct)

        # 상태 변수
        theta_int = 0.0
        theta_enc_prev = counts_to_rad(0)
        t_prev = time.perf_counter()

        print(" θ̂(deg)    θ_enc    θ_int   ω(rad/s)   α")
        # ---------- 루프 ----------
        while True:
            # 1) 센서 읽기
            enc_buf  = array('l', [0])
            tacho_buf= array('d', [0.0])
            card.read_encoder(PEND_CH, 1, enc_buf)
            card.read_other  (TACHO_CH, 1, tacho_buf)

            theta_enc = counts_to_rad(enc_buf[0])
            omega     = tacho_buf[0] * 2 * math.pi / CNT_PER_REV

            # 2) 시간 간격
            t_now = time.perf_counter()
            dt    = t_now - t_prev
            t_prev = t_now

            # 3) 타코 적분각 업데이트
            theta_int += omega * dt

            # 4) 보정된 각도(컴플리멘터리)
            theta_hat, alpha = complementary_filter(theta_enc, theta_int, omega)

            # 5) 출력
            print(f"{math.degrees(theta_hat):8.2f}  "
                  f"{math.degrees(theta_enc):8.2f}  "
                  f"{math.degrees(theta_int):8.2f}  "
                  f"{omega:9.3f}   {alpha:4.2f}")

            # 6) 주기 맞추기
            time.sleep(max(0.0, DT - (time.perf_counter() - t_now)))

    except KeyboardInterrupt:
        print("\n사용자 종료")
    except HILError as e:
        print(f"HIL error {e.error_code}: {e.get_error_message()}")
    finally:
        card.close()


if __name__ == "__main__":
    main()
