import sys
import time
import math
import serial
import cv2
import PySimpleGUI as sg
from util import radian, writeParams, loadParams, spin, spin2, degree, t_all
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
    servo_deg = angle_to_servo(ch, deg)

    set_servo_angle(ch, servo_deg)

def draw_grid(frame):
    h, w, _ = frame.shape

    for y in range(0, h, h // 20):
        cv2.line(frame, (0, y), (w, y), (255, 0, 0))

    for x in range(0, w, w // 20):
        cv2.line(frame, (x, 0), (x, h), (255, 0, 0))

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



def sleep(sec):
    start_time = time.time()
    while time.time() - start_time < sec:
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

    writeParams(params)

down_pos = None
radius = None

def printCoor(event,x,y,flags,param):
    global down_pos, radius

    if event == cv2.EVENT_LBUTTONDOWN:
        print('down', x, y)
        down_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        print('up'  , x, y)
        down_pos = None
        radius   = None

    elif event == cv2.EVENT_MOUSEMOVE and down_pos is not None:
        dx = down_pos[0] - x
        dy = down_pos[1] - y
        radius = int(math.sqrt(dx * dx + dy * dy))

if __name__ == '__main__':

    params, servo_angles, Angles = loadParams()
    marker1, marker2, marker3 = params['markers']

    init_servo(params, servo_angles, Angles)

    initCamera()

    layout = [
        [
            sg.Column([
                [
                    sg.TabGroup([
                        [
                            sg.Tab('marker1', [
                                spin('H lo', '-Hlo1-', marker1['h-lo'], 0, 180, False),
                                spin('H hi', '-Hhi1-', marker1['h-hi'], 0, 180, False),
                                spin('S lo', '-Slo1-', marker1['s-lo'], 0, 255, False),
                                spin('V lo', '-Vlo1-', marker1['v-lo'], 0, 255, False)
                            ])
                            , 
                            sg.Tab('marker2', [
                                spin('H lo', '-Hlo2-', marker2['h-lo'], 0, 180, False),
                                spin('H hi', '-Hhi2-', marker2['h-hi'], 0, 180, False),
                                spin('S lo', '-Slo2-', marker2['s-lo'], 0, 255, False),
                                spin('V lo', '-Vlo2-', marker2['v-lo'], 0, 255, False)
                            ])
                            , 
                            sg.Tab('marker3', [
                                spin('H lo', '-Hlo3-', marker3['h-lo'], 0, 180, False),
                                spin('H hi', '-Hhi3-', marker3['h-hi'], 0, 180, False),
                                spin('S lo', '-Slo3-', marker3['s-lo'], 0, 255, False),
                                spin('V lo', '-Vlo3-', marker3['v-lo'], 0, 255, False)
                            ])
                        ]
                    ], key='-markers-')
                ]
                ,
                [
                    sg.Text('', key='-rotation-')
                ]
            ])
            ,
            sg.Column([
                spin2(f'J{i+1}', f'J{i+1}', servo_angles[i], degree(Angles[i]), -120, 120, True) + [sg.Button('start', key=f'-start-J{i+1}-')]
                for i in range(6)
            ])
        ]
        ,
        [ sg.Checkbox('grid', default=False, key='-show-grid-'), sg.Button('Reset'), sg.Button('Close')]
    ]

    window = sg.Window('Servomotor', layout, disable_minimize=True, element_justification='c')

    start_keys = [ f'-start-J{i+1}-' for i in range(6) ]

    last_capture = time.time()
    is_first = True
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
                writeParams(params)

        jKeys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

        servo_angle_keys = [ f'J{i+1}-servo' for i in range(6) ]

        if event in servo_angle_keys:
            ch = servo_angle_keys.index(event)
            deg = float(values[event])

            moving = move_servo(ch, deg)

            window[jKeys[ch]].update(value=int(servo_to_angle(ch, deg)))

        elif event in jKeys:
            ch = jKeys.index(event)
            deg = float(values[event])

            moving = move_joint(ch, deg)

            window[servo_angle_keys[ch]].update(value=int(angle_to_servo(ch, deg)))

        elif event in start_keys:
            moving = calibrate_angle(event)

        elif event == 'Reset':
            for ch in range(6):
                set_servo_angle(ch, 90)
                window[f'J{ch + 1}-servo'].update(value=90)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            writeParams(params)

            closeCamera()
            break

        elif event == "Calibrate":
            moving = Calibrate(params, values)        

        else:
            if 0.1 < time.time() - last_capture:
                last_capture = time.time()

                frame = getCameraFrame()

                if radius is None:
                    frame, marker_deg = get_markers(window, values, frame)

                else:
                    marker_idx = [ 'marker1', 'marker2', 'marker3' ].index( window['-markers-'].get() )

                    frame = set_hsv_range(params, window, marker_idx, frame, down_pos, radius)

                if values['-show-grid-']:
                    draw_grid(frame)

                cv2.imshow("camera", frame)

                if is_first:

                    is_first = False
                
                    cv2.setMouseCallback('camera', printCoor)

    window.close()

