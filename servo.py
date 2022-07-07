import sys
import time
import math
import serial
import cv2
import PySimpleGUI as sg
from util import nax, read_params, servo_angle_keys, radian, write_params, spin, spin2, degree, t_all, sleep
import numpy as np


servo_angles = [np.nan] * nax

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


def init_servo(params_arg):
    global params, servo_angles, servo_param, ser

    params = params_arg

    for ch, deg in enumerate(params['servo-angles']):
        servo_angles[ch] = deg

    com_port = params['COM'] 
    servo_param = params['calibration']['servo']

    try:
        ser = serial.Serial(com_port, 115200, timeout=1, write_timeout=1)
    except serial.serialutil.SerialException: 
        print(f'指定されたシリアルポートがありません。{com_port}')
        sys.exit(0)

def set_servo_param(ch, scale, offset):
    servo_param[ch] = [scale, offset]
    params['calibration']['servo'][ch] = [scale, offset]

    write_params(params)

def angle_to_servo(ch, deg):
    coef, intercept = servo_param[ch]

    return coef * deg + intercept

def servo_to_angle(ch, deg):
    coef, intercept = servo_param[ch]

    return (deg - intercept) / coef

def set_servo_angle(ch : int, deg : float):
    servo_angles[ch] = deg

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

def move_servo(ch, dst):
    src = servo_angles[ch]

    start_time = time.time()
    while True:
        t = (time.time() - start_time) / t_all
        if 1 <= t:
            break

        deg = t * dst + (1 - t) * src

        set_servo_angle(ch, deg)

        yield

def move_all_servo(dsts):
    srcs = list(servo_angles)

    start_time = time.time()
    while True:
        t = (time.time() - start_time) / t_all
        if 1 <= t:
            break

        for ch in range(nax):

            deg = t * dsts[ch] + (1 - t) * srcs[ch]

            set_servo_angle(ch, deg)

        yield

if __name__ == '__main__':

    params = read_params()

    init_servo(params)

    layout = [
        [
            spin(f'J{i+1}', f'J{i+1}-servo', int(servo_angles[i]), -120, 120, True) for i in range(nax)
        ]
        +
        [ sg.Button('Ready'), sg.Button('Close') ]
    ]

    window = sg.Window('Servomotor', layout, element_justification='c') # disable_minimize=True, 

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
            deg = values[event]

            moving = move_servo(ch, deg)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            write_params(params)

            break

    window.close()

