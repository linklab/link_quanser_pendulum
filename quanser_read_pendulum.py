import math, time
from array import array
from quanser.hardware import HIL, HILError, MAX_STRING_LENGTH

# ---- 하드웨어/펌듈럼 기본 매개변수 ----
PEND_CH          = array('I', [1])      # Qube‑Servo 3: 펜듈럼 인코더 = 채널 1
TACH_CH  = array('I', [14001])
ENC_RESOLUTION   = 2048                 # counts / rev  (사양표)
PWM_SCALE        = 0.0                  # 초기화 루프에선 모터 OFF
DT               = 0.001                 # 10 ms  (Windows라면 10 ms 주기 정도가 현실적)
enc_val = array('i', [0])
tach_val = array('d', [0.0])

def counts_to_rad(counts: int) -> float:
    """인코더 카운트를 라디안으로 변환"""
    return counts * 2.0*math.pi / ENC_RESOLUTION

# ------------------------------------------------------------------
def init_card() -> HIL:
    """HIL 객체 열고 PWM 모드 활성화"""
    card = HIL("qube_servo3_usb", "0")
    # PWM 하드웨어를 쓰려면 카드 옵션을 열어 둔다
    card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)
    return card

def zero_pendulum_encoder(card: HIL) -> None:
    """펜듈럼을 ‘현재 위치 = 0 rad’ 로 영점 설정"""
    zero_ct = array('l', [0])
    card.set_encoder_counts(PEND_CH, len(PEND_CH), zero_ct)

def read_pendulum(card: HIL,
                  prev_angle: float | None,
                  prev_time: float | None) -> tuple[float, float, float]:
    """
    펜듈럼 각도(rad), 각속도(rad/s), 측정 시각(s)를 반환
    prev_* 가 None이면 각속도 = 0
    """
    card.read_encoder(PEND_CH, len(PEND_CH), enc_val)
    angle = counts_to_rad(enc_val[0])

    now = time.perf_counter()
    if prev_angle is None:
        omega = 0.0
    else:
        omega = (angle - prev_angle) / (now - prev_time)
    return angle, omega, now

# ------------------------------------------------------------------
def main():
    try:
        card = init_card()
        print("카드 열림 – PWM 활성화 완료")

        # 1) 펜듈럼 영점 잡기 (필요하면 손으로 6 시 방향에 둔 뒤)
        print("펜듈럼 영점(6 시) 설정…")
        zero_pendulum_encoder(card)
        time.sleep(1.0)  # 영점 잡는 동안 잠시 대기
        
        prev_angle = prev_time = None

        # 2) 읽기 루프
        print("---- 각도 / 각속도 측정 시작 ----")
        a = 0
        while True:
            a += 1
            angle, omega, now = read_pendulum(card, prev_angle, prev_time)
            deg = math.degrees(angle)
            card.read_other(TACH_CH, 1, tach_val)
            print(f"θ = {deg%360:8.3f} deg   ω = {omega:8.3f} rad/s   raw cnt = {enc_val}")
            tach_vel_rad = tach_val[0] * 2.0 * math.pi / 2048.0
            # print("abs(omega - tach_vel_rad):", abs(omega - tach_vel_rad),)

            prev_angle, prev_time = angle, now
            time.sleep(DT)
            # if a % 500 == 0:
            #     card.close()
            #     card = init_card()
                # print("펜듈럼 영점 재설정…")
                # zero_pendulum_encoder(card)
                # time.sleep(1)

    except HILError as e:
        # Quanser API 호출 실패 시
        print(f"HIL Error {e.error_code}: {e.get_error_message()}")

    except KeyboardInterrupt:
        print("\n측정 중단(사용자)")

    finally:
        # 안전하게 모터 정지 & 카드 닫기
        try:
            card.write_pwm(array('I', [0]), 1, array('d', [0.0]))
        except Exception:
            pass
        card.close()
        print("카드 연결 종료")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
