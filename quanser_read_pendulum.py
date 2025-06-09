from quanser.hardware import HIL
from array   import array
import math, time

card = HIL("qube_servo3_usb", "0")
try:
    enc_ch  = array('I', [1])
    enc_val = array('l', [0])

    # 각도가 안 맞을 시 펜듈럼 각도 리셋 시 사용
    # zero_ct = array('l', [0])
    # card.set_encoder_counts(enc_ch, len(enc_ch), zero_ct)

    while True:
        card.read_encoder(enc_ch, 1, enc_val)

        alpha = enc_val[0] * 2*math.pi / 2048.0
        alpha_deg = alpha * 180/math.pi

        print(f"pendulum α = {alpha_deg} °")
        time.sleep(0.5)

finally:
    card.close()