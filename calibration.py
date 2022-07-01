import time
import math
import numpy as np
import PySimpleGUI as sg
import json
import cv2
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand, getCameraFrame
from util import nax, jKeys, read_params, servo_angle_keys, getGlb, radian, write_params, t_all, spin, spin2, degree, Vec2, sleep, loadParams
from servo import j_range, init_servo, set_servo_angle, move_servo
from kinematics import forward_kinematics, inverse_kinematics

def angle_to_servo(ch, deg):
    coef, intercept = servo_param[ch]

    return coef * deg + intercept

def servo_to_angle(ch, deg):
    coef, intercept = servo_param[ch]

    return (deg - intercept) / coef

def set_angle(ch : int, deg : float):
    if not (j_range[ch][0] - 10 <= deg and deg <= j_range[ch][1] + 10):

        print(f'set angle err: ch:{ch} deg:{deg}')
        assert False

    servo_deg = angle_to_servo(ch, deg)

    set_servo_angle(ch, servo_deg)

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


def move_linear(dst):
    global move_linear_ok

    move_linear_ok = True

    src = forward_kinematics(Angles)

    with open('ik.csv', 'w') as f:
        f.write('time,J1,J2,J3,J4,J5,J6\n')
        start_time = time.time()
        while True:
            t = time.time() - start_time
            if t_all <= t:
                break

            r = t / t_all

            pose = [ r * d + (1 - r) * s for s, d in zip(src, dst) ]

            rads = inverse_kinematics(pose)
            if rads is None:

                move_linear_ok = False
            else:
                degs = degree(rads)
                f.write(f'{t},{",".join(["%.1f" % x for x in degs])}\n')

                for ch, deg in enumerate(degs):
                    set_angle(ch, deg)

            yield

def move_xyz(x, y, z):
    x_min = 100
    x_max = 300

    pitch_min = 90
    pitch_max = 45

    assert x_min <= x and x <= x_max

    r = (x - x_min) / (x_max - x_min)

    pitch = r * pitch_max + (1 - r) * pitch_min

    pose = [ x, y, z, 0, radian(pitch)]

    for _ in move_linear(pose):
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


def get_markers(window, values, frame):
    
    centers = [ showMark(values, frame, i) for i in range(3) ]

    if all(x is not None for x in centers):
        v1 = (centers[1] - centers[0]).unit()
        v2 = (centers[2] - centers[1]).unit()

        ip = max(-1, min(1, v1.dot(v2)))
        rad = math.acos(ip)
        assert 0 <= rad and rad <= math.pi

        marker_deg = degree(rad)
        if v1.cross(v2) < 0:
            marker_deg = -marker_deg

        window['-rotation-'].update(value='%.1f' % marker_deg)
    else:
        marker_deg = None
        window['-rotation-'].update(value='')

    return frame, marker_deg



def fitRegression(eye_xy, hand_x, hand_y):
    X = np.array(eye_xy)
    hand_x = np.array(hand_x)
    hand_y = np.array(hand_y)

    glb = getGlb()
    glb.regX = LinearRegression().fit(X, hand_x)

    glb.regY = LinearRegression().fit(X, hand_y)

    print('reg x', glb.regX.coef_)
    print('reg y', glb.regY.coef_)

    prd_x = glb.regX.predict(X)
    prd_y = glb.regY.predict(X)
    print(f'X:{X.shape} arm-x:{hand_x.shape} arm-y:{hand_y.shape} prd-x:{prd_x.shape} prd-y:{prd_y.shape}')

    for i in range(X.shape[0]):
        print(f'cam:{X[i, 0]} {X[i, 1]} arm:{hand_x[i]} {hand_y[i]} prd:{int(prd_x[i]) - hand_x[i]} {int(prd_y[i]) - hand_y[i]}')



def Calibrate(params, values):
    eye_xy = []
    hand_x = []
    hand_y = []

    for x in [150, 200, 250, 300]:
        for y in [-50, 0, 50]:
            print(f'start move x:{x} y:{y}')
            
            for _ in move_xyz(x, y, 30):
                yield

            if not move_linear_ok:
                print(f'skip move x:{x} y:{y}')
                continue
            
            print("move end")
            for _ in sleep(4):
                yield

            centers = [ showMark(values, frame, i) for i in range(2) ]

            (rx, ry), (bx, by) = get_markers(values)

            if bx is not None and rx is not None:
                CamX = 0.5 * (bx + rx)
                CamY = 0.5 * (by + ry)


            print(f'hand eye x:{x} y:{y} cx:{camX()} cy:{camY()}')

            eye_xy.append([ camX(), camY() ])
            hand_x.append(x)
            hand_y.append(y)

    params['calibration'] = {
        'eye-xy' : eye_xy,
        'hand-x' : hand_x,
        'hand-y' : hand_y
    }

    write_params(params)

    fitRegression(eye_xy, hand_x, hand_y)


def draw_grid(frame):
    h, w, _ = frame.shape

    for y in range(0, h, h // 20):
        cv2.line(frame, (0, y), (w, y), (255, 0, 0))

    for x in range(0, w, w // 20):
        cv2.line(frame, (x, 0), (x, h), (255, 0, 0))

down_pos = None
radius = None

def mouse_callback(event,x,y,flags,param):
    global down_pos, radius

    if event == cv2.EVENT_LBUTTONDOWN:
        down_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        down_pos = None
        radius   = None

    elif event == cv2.EVENT_MOUSEMOVE and down_pos is not None:
        dx = down_pos[0] - x
        dy = down_pos[1] - y
        radius = int(math.sqrt(dx * dx + dy * dy))

if __name__ == '__main__':

    params = read_params()

    servo_angles = params['servo-angles']
    servo_param = params['calibration']['servo']

    init_servo(params, servo_angles)

    initCamera()

    layout = [
        [
            sg.Column([
                [
                    sg.Text('', key='-rotation-')
                ]
            ])
            ,
            sg.Column([
                spin2(f'J{ch+1}', f'J{ch+1}', deg, servo_to_angle(ch, deg), -120, 120, True) + [sg.Button('start', key=f'-start-J{ch+1}-')]
                for ch, deg in enumerate(servo_angles)
            ])
        ]
        ,
        [ sg.Checkbox('grid', default=False, key='-show-grid-'), sg.Button('Reset'), sg.Button('Close')]
    ]

    window = sg.Window('calibration', layout, disable_minimize=True, element_justification='c')

    start_keys = [ f'-start-J{i+1}-' for i in range(nax) ]

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
                write_params(params)

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
            for ch in range(nax):
                set_servo_angle(ch, 90)
                window[f'J{ch + 1}-servo'].update(value=90)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            write_params(params)

            closeCamera()
            break

        elif event == "Calibrate":
            moving = Calibrate(params, values)        

        else:
            if 0.1 < time.time() - last_capture:
                last_capture = time.time()

                frame = getCameraFrame()

                if values['-show-grid-']:
                    draw_grid(frame)

                cv2.imshow("camera", frame)

                if is_first:

                    is_first = False
                
                    cv2.setMouseCallback('camera', mouse_callback)
