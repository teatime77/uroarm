import sys
import time
import math
import serial
import cv2
import PySimpleGUI as sg
from util import nax, servo_angle_keys, radian, write_params, loadParams, spin, spin2, degree, t_all, sleep
from camera import initCamera, getCameraFrame, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand
from calibration import Calibrate, get_markers, set_hsv_range
import numpy as np
from sklearn.linear_model import LinearRegression

j_range = [
    [  -80,  80 ],
    [ -100, -30 ],
    [   50, 120 ],
    [   25,  70 ],
    [  -90,  90 ],
    [    0,  90 ]
]

def init_servo_nano(params):
    global pwm
    try:
        import Adafruit_PCA9685

        pwm = Adafruit_PCA9685.PCA9685(address=0x40)
        pwm.set_pwm_freq(60)

    except ModuleNotFoundError:
        print("no Adafruit")

def setPWM(ch, pos):
    pulse = round( 150 + (600 - 150) * (pos + 0.5 * math.pi) / math.pi )
    pwm.set_pwm(ch, 0, pulse)

def set_servo_angle_nano(ch, deg):
    rad = radian(deg)
    setPWM(ch, rad)


def init_servo(params, servo_angles_arg, angles_arg):
    global servo_angles, Angles, servo_param, ser

    servo_angles = servo_angles_arg

    Angles   = angles_arg
    com_port = params['COM'] 

    servo_param = params['calibration']['servo']

    try:
        ser = serial.Serial(com_port, 115200, timeout=1, write_timeout=1)
    except serial.serialutil.SerialException: 
        print(f'指定されたシリアルポートがありません。{com_port}')
        sys.exit(0)

def angle_to_servo(ch, deg):
    coef, intercept = servo_param[ch]

    return coef * deg + intercept

def servo_to_angle(ch, deg):
    coef, intercept = servo_param[ch]

    return (deg - intercept) / coef

def set_servo_angle(ch : int, deg : float):
    servo_angles[ch] = deg
    Angles[ch] = radian(servo_to_angle(ch, deg))

    cmd = "%d,%.1f\r" % (ch, deg)

    while True:
        try:
            n = ser.write(cmd.encode('utf-8'))
            break
        except serial.SerialTimeoutException:
            print("write time out")
            time.sleep(1)

    ret = ser.read_all().decode('utf-8')
    if "error" in ret:
        print("read", ret)

def set_angle(ch : int, deg : float):
    if not (j_range[ch][0] - 10 <= deg and deg <= j_range[ch][1] + 10):

        print(f'set angle err: ch:{ch} deg:{deg}')
        assert False

    servo_deg = angle_to_servo(ch, deg)

    set_servo_angle(ch, servo_deg)

def move_servo(ch, dst):
    src = servo_angles[ch]

    start_time = time.time()
    while True:
        total_time = t_all
        t = (time.time() - start_time) / total_time
        if 1 <= t:
            break

        deg = t * dst + (1 - t) * src

        set_servo_angle(ch, deg)

        yield

def move_joint(ch, dst):
    src = servo_to_angle(ch, servo_angles[ch])

    start_time = time.time()
    while True:
        total_time = t_all
        t = (time.time() - start_time) / total_time
        if 1 <= t:
            break

        deg = t * dst + (1 - t) * src

        set_angle(ch, deg)

        yield

def move_all_joints(dsts):
    srcs = [ servo_to_angle(ch, servo_angles[ch]) for ch in range(nax) ]

    start_time = time.time()
    while True:
        total_time = t_all
        t = (time.time() - start_time) / total_time
        if 1 <= t:
            break

        for ch in range(nax):

            deg = t * dsts[ch] + (1 - t) * srcs[ch]

            set_angle(ch, deg)

        yield


def calibrate_angle(event):
    global marker_deg

    ch = start_keys.index(event)

    min_deg, max_deg = j_range[ch]

    dev_deg = servo_angles[ch]

    cnt = 5
    targets = []
    dev_degs = []
    for idx in range(cnt + 1):
        target = min_deg + (max_deg - min_deg) * idx / cnt

        print(f'idx:{idx} target:{target}')
        ok_cnt = 0
        for i in range(10000):
            if marker_deg is None:
                yield
                continue

            diff = target - marker_deg
            marker_deg = None

            if abs(diff) < 0.5:
                ok_cnt += 1
                if ok_cnt < 5:
                    yield
                    continue
                else:
                    targets.append(target)
                    dev_degs.append(dev_deg)

                    print(f'idx:{idx} done {i}')
                    break

            else:
                ok_cnt = 0

            if ch == 1:
                dev_deg += - np.sign(diff) * 0.1
            else:
                dev_deg += np.sign(diff) * 0.1

            set_servo_angle(ch, dev_deg)

            for _ in sleep(0.05):
                yield

            yield

    X = np.array(targets).reshape(-1, 1) 
    Y = np.array(dev_degs)

    reg = LinearRegression().fit(X, Y)
    print('reg x', reg.coef_, reg.intercept_)
    prd_x = reg.predict(X)

    with open('data/angle.csv', 'w') as f:
        f.write('target, dev, prd, reg\n')

        for target, dev_deg, prd_deg in zip(targets, dev_degs, prd_x.tolist()):

            f.write(f'{target}, {dev_deg}, {prd_deg}, {reg.coef_[0] * target + reg.intercept_}\n')

    params['calibration']['servo'][ch] = [ reg.coef_[0], reg.intercept_ ]

    write_params(params)

if __name__ == '__main__':

    params, servo_angles, Angles = loadParams()

    init_servo(params, servo_angles, Angles)

    layout = [
        [ sg.Button('Close') ]
    ]

    window = sg.Window('Servomotor', layout, disable_minimize=True, element_justification='c')

    moving = None

    while True:
        event, values = window.read(timeout=1)

        if moving is not None:
            try:
                moving.__next__()

            except StopIteration:
                moving = None
                print('========== stop moving ==========')

                params['servo-angles'] = servo_angles
                write_params(params)

        if event in servo_angle_keys:
            ch = servo_angle_keys.index(event)
            deg = float(values[event])

            moving = move_servo(ch, deg)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            write_params(params)

            break

    window.close()

