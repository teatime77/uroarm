import sys
import time
import math
import serial
import cv2
import PySimpleGUI as sg
from util import nax, jKeys, read_params, servo_angle_keys, radian, write_params, spin, spin2, degree, t_all, sleep
from servo import init_servo, set_servo_angle, move_servo, move_all_servo, angle_to_servo, servo_to_angle, set_servo_param, servo_angles
from camera import initCamera, closeCamera, getCameraFrame
import numpy as np


def spin3(ch, label, key, val1, val2, min_val, max_val, a, b, bind_return_key = True):

    return [ 
        sg.Text(label, size=(5,1)),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=int(val1), size=(5, 1), key=key+'-servo', enable_events=not bind_return_key, bind_return_key=bind_return_key ),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=int(val2), size=(5, 1), key=key, enable_events=not bind_return_key, bind_return_key=bind_return_key ),
        sg.Button(f'{a}', key=f'-datum-angle-{ch}-0-', size=(5,1)), 
        sg.Button(f'{b}', key=f'-datum-angle-{ch}-1-', size=(5,1))
    ]

def draw_grid(frame):
    h, w, _ = frame.shape

    for y in range(0, h, h // 20):
        cv2.line(frame, (0, y), (w, y), (255, 0, 0))

    for x in range(0, w, w // 20):
        cv2.line(frame, (x, 0), (x, h), (255, 0, 0))

    cx, cy = w // 2, h // 2
    r = math.sqrt(cx * cx + cy * cy)

    x1, y1 = r * np.cos(radian(30)), r * np.sin(radian(30)), 
    cv2.line(frame, (int(cx - x1), int(cy + y1)), (int(cx + x1), int(cy - y1)), (255, 0, 0))
    cv2.line(frame, (int(cx - x1), int(cy - y1)), (int(cx + x1), int(cy + y1)), (255, 0, 0))
    
    x1, y1 = r * np.cos(radian(60)), r * np.sin(radian(60)), 
    cv2.line(frame, (int(cx - x1), int(cy + y1)), (int(cx + x1), int(cy - y1)), (255, 0, 0))
    cv2.line(frame, (int(cx - x1), int(cy - y1)), (int(cx + x1), int(cy + y1)), (255, 0, 0))

def calibrate_angle(event):
    ch, idx = [ int(x) for x in event.split('-')[3:5] ]

    datum_servo_angles[ch, idx] = servo_angles[ch]
    window[event].update(button_color=('white', 'blue'))

    servo1, servo2 = datum_servo_angles[ch, :]
    if np.isnan(servo1) or np.isnan(servo2):
        return

    angle1, angle2 = datum_angles[ch]

    # scale * angle1 + offset = servo1
    # scale * angle2 + offset = servo2

    offset = (angle2 * servo1 - angle1 * servo2) / (angle2 - angle1)
    if abs(angle1) < abs(angle2):
        scale = (servo2 - offset) / angle2
    else:
        scale = (servo1 - offset) / angle1

    set_servo_param(ch, scale, offset)

    window[jKeys[ch]].update(value=int(servo_to_angle(ch, servo_angles[ch])))

if __name__ == '__main__':

    params = read_params()

    datum_angles = params['datum-angles']

    datum_servo_angles = np.full((nax, 2), np.nan)

    init_servo(params)
    initCamera()

    layout = [
        [
            [ sg.Text('joint', size=(5,1)), sg.Text('servo', size=(5,1)), sg.Text('arm', size=(5,1)), sg.Text('min', size=(5,1)), sg.Text('max', size=(5,1)) ]
        ]
        +
        [
            spin3(ch, f'J{ch+1}', f'J{ch+1}', deg, servo_to_angle(ch, deg), -120, 150, a, b, True) 
                for ch, (deg, (a, b)) in enumerate(zip(servo_angles, datum_angles))
        ]
        +
        [
            [ sg.Button('Close') ]            
        ]
    ]

    window = sg.Window('angle calibration', layout, element_justification='c') # disable_minimize=True, 

    moving = None
    last_capture = time.time()

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

        elif event.startswith('-datum-angle-'):
            calibrate_angle(event)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            write_params(params)

            closeCamera()
            break

        else:
            if 0.1 < time.time() - last_capture:
                last_capture = time.time()

                frame = getCameraFrame()

                draw_grid(frame)

                cv2.imshow("camera", frame)

    window.close()

