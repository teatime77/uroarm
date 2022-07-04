import sys
import time
import math
import numpy as np
import PySimpleGUI as sg
import json
import cv2
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand, getCameraFrame
from util import nax, jKeys, pose_keys, read_params, servo_angle_keys, getGlb, radian, write_params, t_all, spin, spin2, degree, Vec2, Vec3, sleep, get_pose, show_pose
from servo import j_range, init_servo, set_servo_angle, move_servo, move_all_servo, angle_to_servo, servo_to_angle, set_servo_param, servo_angles
from kinematics import forward_kinematics, inverse_kinematics
from marker import init_markers, detect_markers

rect_pose = [0, ]
def set_angle(ch : int, deg : float):
    if not (j_range[ch][0] - 10 <= deg and deg <= j_range[ch][1] + 10):

        # print(f'set angle err: ch:{ch} deg:{deg}')
        # assert False
        pass

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

    src = forward_kinematics(servo_angles)

    with open('ik.csv', 'w') as f:
        f.write('time,J1,J2,J3,J4,J5,J6\n')
        start_time = time.time()
        while True:
            t = time.time() - start_time
            if t_all <= t:
                break

            r = t / t_all

            pose = [ r * d + (1 - r) * s for s, d in zip(src, dst) ]

            rad5s = inverse_kinematics(pose)
            if rad5s is None:

                move_linear_ok = False
            else:
                deg5s = degree(rad5s) 
                f.write(f'{t},{",".join(["%.1f" % x for x in deg5s])}\n')

                for ch, deg in enumerate(deg5s):
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

    yaw = - math.atan2(y, x)
    pose = [ x, y, z, yaw, radian(pitch)]

    for _ in move_linear(pose):
        yield

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



def Calibrate():
    eye_xy = []
    hand_x = []
    hand_y = []

    for y in [-50, 0, 50]:
        for x in [150, 200, 250, 300, 150]:
            print(f'start move x:{x} y:{y}')
            
            for _ in move_xyz(x, y, 30):
                yield

            if not move_linear_ok:
                print(f'skip move x:{x} y:{y}')
                continue
            
            print("move end")
            for _ in sleep(4):
                yield


            # eye_xy.append([ camX(), camY() ])
            # hand_x.append(x)
            # hand_y.append(y)

    # params['calibration'] = {
    #     'eye-xy' : eye_xy,
    #     'hand-x' : hand_x,
    #     'hand-y' : hand_y
    # }

    # write_params(params)

    # fitRegression(eye_xy, hand_x, hand_y)


def draw_grid(frame):
    h, w, _ = frame.shape

    for y in range(0, h, h // 20):
        cv2.line(frame, (0, y), (w, y), (255, 0, 0))

    for x in range(0, w, w // 20):
        cv2.line(frame, (x, 0), (x, h), (255, 0, 0))

    cx, cy = w // 2, h // 2
    r = math.sqrt(cx * cx + cy * cy)

    x1, y1 = r * np.cos(radian(30)), r * np.sin(radian(30)), 
    cv2.line(frame, (int(cx - x1), int(cy + y1)), (int(cx + x1), int(cy - y1)), (255, 0, 0), thickness=2)
    cv2.line(frame, (int(cx - x1), int(cy - y1)), (int(cx + x1), int(cy + y1)), (255, 0, 0), thickness=2)
    
    x1, y1 = r * np.cos(radian(60)), r * np.sin(radian(60)), 
    cv2.line(frame, (int(cx - x1), int(cy + y1)), (int(cx + x1), int(cy - y1)), (255, 0, 0), thickness=2)
    cv2.line(frame, (int(cx - x1), int(cy - y1)), (int(cx + x1), int(cy + y1)), (255, 0, 0), thickness=2)

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

def angle_from_3points(points):
    assert len(points) == 3

    v1 = (points[1] - points[0]).unit()
    v2 = (points[2] - points[1]).unit()

    ip = max(-1, min(1, v1.dot(v2)))
    rad = math.acos(ip)
    assert 0 <= rad and rad <= math.pi

    marker_deg = degree(rad)
    if v1.cross(v2) < 0:
        marker_deg = -marker_deg

    return marker_deg

def show_next_pose(ch, servo_deg):
    degs = list(servo_angles)
    degs[ch] = servo_deg

    pose = forward_kinematics(degs)            
    show_pose(window, pose)

def hand_eye_calibration(marker_table):
    if(np.isnan(marker_table[6:, 0]).any()):
        return np.nan

    marker_table[:, 2] = - marker_table[:, 2]

    p1, p2, p3, p4 = [ Vec3(* marker_table[i, :3].tolist()) for i in range(4) ]

    normal_vec = (p2 - p1).cross(p3 - p1).unit()

    h = normal_vec.dot(p4 - p1)

    return h


if __name__ == '__main__':

    params = read_params()

    marker_ids = params['marker-ids']
    datum_angles = params['datum-angles']

    datum_servo_angles = np.full((nax, 2), np.nan)

    init_servo(params)
    init_markers(params)
    initCamera()


    if len(sys.argv) == 2:
        from infer import Inference

        inference = Inference()

    else:
        inference = None

    marker_table = np.array([[0] * 6] * 10, dtype=np.float32)
    layout = [
        [
            sg.Column([
                [
                    sg.Text('', key='-tcp-height-')
                ]
            ])
            ,
            sg.Column([
                spin2(f'J{ch+1}', f'J{ch+1}', deg, servo_to_angle(ch, deg), -120, 120, True) + [ 
                    sg.Text('', key=f'-yaw-{ch+1}-'), 
                    sg.Text('', key=f'-angle-{ch+1}-'), 
                    sg.Text('', key=f'-vec-{ch+1}-')
                ]
                for ch, deg in enumerate(servo_angles)
            ])
            ,
            sg.Column([
                [ sg.Button(f'{a}', key=f'-datum-angle-{ch}-0-', size=(4,1)), sg.Button(f'{b}', key=f'-datum-angle-{ch}-1-', size=(4,1)) ] for ch, (a, b) in enumerate(datum_angles) 
            ])
            ,
            sg.Column([
                spin('X', 'X' , 0,    0, 400 ),
                spin('Y', 'Y' , 0, -300, 300 ),
                spin('Z', 'Z' , 0,    0, 150 ),
                spin('R1', 'R1', 0, -90,  90 ),
                spin('R2', 'R2', 0,   0, 120 )
            ])
            ,
            sg.Table(marker_table.tolist(), headings=['x', 'y', 'z', 'yaw', 'angle1', 'angle2'], auto_size_columns=False, col_widths=[6]*6, num_rows=10, key='-marker-table-')
        ]
        ,
        [ sg.Checkbox('grid', default=False, key='-show-grid-'), sg.Button('Ready'), sg.Button('Calibrate'), sg.Button('Close')]
    ]

    window = sg.Window('calibration', layout, element_justification='c', finalize=True) # disable_minimize=True

    pose = forward_kinematics(servo_angles)
    show_pose(window, pose)

    last_capture = time.time()
    is_first = True

    moving = None

    while True:
        event, values = window.read(timeout=1)

        if moving is not None:
            try:
                moving.__next__()
                continue

            except StopIteration:
                moving = None
                print('========== stop moving ==========')

                params['servo-angles'] = servo_angles
                write_params(params)

        if event in servo_angle_keys:
            ch = servo_angle_keys.index(event)
            deg = float(values[event])

            show_next_pose(ch, deg)

            moving = move_servo(ch, deg)

            window[jKeys[ch]].update(value=int(servo_to_angle(ch, deg)))

        elif event in jKeys:
            ch = jKeys.index(event)
            deg = float(values[event])

            servo_deg = angle_to_servo(ch, deg)
            show_next_pose(ch, servo_deg)

            moving = move_joint(ch, deg)

            window[servo_angle_keys[ch]].update(value=int(servo_deg))

        elif event in pose_keys:
            pose = get_pose(values)
            moving = move_linear(pose)

        elif event.startswith('-datum-angle-'):
            calibrate_angle(event)

        elif event == 'Ready':
            degs = params['ready']
            for ch, deg in enumerate(degs):
                window[jKeys[ch]].update(value=deg)

            moving = move_all_joints(degs)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            write_params(params)

            closeCamera()
            break

        elif event == "Calibrate":
            moving = Calibrate()        

        else:
            if 0.1 < time.time() - last_capture:
                last_capture = time.time()

                frame = getCameraFrame()

                cx, cy = np.nan, np.nan
                if inference is not None:
                    cx, cy = inference.get(frame)

                frame, vecs = detect_markers(marker_ids, frame)

                for ch, vec in enumerate(vecs):                    
                    if vec is None:

                        marker_table[ch, :3] = [np.nan] * 3
                    else:

                        ivec = [int(x) for x in vec]

                        marker_table[ch, :3] = ivec[:3]

                yaws = [np.nan if vec is None else vec[5] for vec in vecs]

                angles1 = [np.nan] * len(marker_ids)
                for i, (ch1, ch2) in enumerate([ [0,1], [1,3], [3, 4] ]):
                    angles1[i+1] = np.round(yaws[ch2] - yaws[ch1])

                angles2 = [np.nan] * len(marker_ids)
                for ch in range(1, len(vecs) - 1):
                    if all(v is not None for v in vecs[ch-1:ch+2] ):
                        points = [ Vec2(v[0], v[1]) for v in vecs[ch-1:ch+2] ]
                        angles2[ch] = int(angle_from_3points(points))

                marker_table[:, 3] = yaws
                marker_table[:, 4] = angles1
                marker_table[:, 5] = angles2

                h = hand_eye_calibration(marker_table[6:, :])
                window['-tcp-height-'].update(value=f'height:{h:.1f}')

                window['-marker-table-'].update(values=marker_table.tolist())


                if values['-show-grid-']:
                    draw_grid(frame)


                if not np.isnan(cx):
                    cv2.circle(frame, (int(cx), int(cy)), 10, (255,255,255), -1)

                cv2.imshow("camera", frame)

                window.refresh()

                if is_first:

                    is_first = False
                
                    cv2.setMouseCallback('camera', mouse_callback)
