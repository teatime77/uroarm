import sys
import time
import math
import numpy as np
import PySimpleGUI as sg
import json
import cv2
from sklearn.linear_model import LinearRegression
from camera import initCamera, closeCamera, getCameraFrame
from util import nax, jKeys, pose_keys, read_params, servo_angle_keys, radian, write_params, t_all, spin, spin2, degree, Vec2, Vec3, sleep, get_pose, show_pose
from servo import init_servo, set_servo_angle, move_servo, move_all_servo, angle_to_servo, servo_to_angle, set_servo_param, servo_angles
from kinematics import forward_kinematics, inverse_kinematics
from marker import init_markers, detect_markers

hand_idx = nax - 1

PICK_Z = 10
LIFT_Z = 30

pose1 = [ 90, -90, 50, math.pi / 4, math.pi / 2 ]


rect_pose = [0, ]
def set_angle(ch : int, deg : float):
    servo_deg = angle_to_servo(ch, deg)

    set_servo_angle(ch, servo_deg)

def move_joint(ch, dst):
    global is_moving

    is_moving = True

    src = servo_to_angle(ch, servo_angles[ch])

    start_time = time.time()
    while True:
        t = (time.time() - start_time) / t_all
        if 1 <= t:
            break

        deg = t * dst + (1 - t) * src

        set_angle(ch, deg)

        yield

    is_moving = False

def move_all_joints(dsts):
    global is_moving

    is_moving = True

    srcs = [ servo_to_angle(ch, servo_angles[ch]) for ch in range(nax) ]

    start_time = time.time()
    while True:
        t = (time.time() - start_time) / t_all
        if 1 <= t:
            break

        for ch in range(nax):

            deg = t * dsts[ch] + (1 - t) * srcs[ch]

            set_angle(ch, deg)

        yield

    is_moving = False

def open_hand():
    for _ in move_joint(hand_idx, 50):
        yield

def close_hand():
    for _ in move_joint(hand_idx, 20):
        yield

def move_linear(dst):
    global move_linear_ok, is_moving

    is_moving = True
    move_linear_ok = True

    src = forward_kinematics(servo_angles)

    with open('data/ik.csv', 'w') as f:
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

    is_moving = False

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

def move_to_ready():
    degs = params['ready']
    for ch, deg in enumerate(degs):
        window[jKeys[ch]].update(value=deg)

    for _ in move_all_joints(degs):
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

def calibrate_xy():
    tcp_scrs = []
    arm_xyz = []

    ys = [-80, 0, 80]
    xs = [120, 190, 250]
    # ys = [-60, 60]
    # xs = [150, 250]

    f = open('data/calibrate-xy.csv', 'w')
    f.write('scr-x, scr-y, tcp-height, arm-x, arm-y, arm-z, prd-arm-x, prd-arm-y\n')

    tcp_heights = []
    for arm_y in ys:
        for arm_x in xs:
            print(f'start move x:{arm_x} y:{arm_y}')

            # z = LIFT_Zの位置に移動する。
            arm_z = LIFT_Z
            for _ in move_xyz(arm_x, arm_y, arm_z):
                yield
            
            print("move xy end")
            for _ in sleep(3):
                yield

            for trial in range(10000):
                while np.isnan(tcp_height):
                    yield

                diff = tcp_height - PICK_Z
                print(f'move z trial:{trial} diff:{diff:.1f}')
                if abs(diff) < 5:
                    break

                # PICK_Zの位置に移動する。
                arm_z -= tcp_height - PICK_Z
                for _ in move_xyz(arm_x, arm_y, arm_z):
                    yield

            for _ in sleep(3):
                yield

            print(f'move z end:{tcp_height:.1f}')

            while np.isnan(tcp_height):
                yield

            tcp_heights.append(tcp_height)

            arm_xyz.append([arm_x, arm_y, arm_z])
            tcp_scrs.append([tcp_scr.x, tcp_scr.y])

            # z = LIFT_Zの位置に移動する。
            for _ in move_xyz(arm_x, arm_y, LIFT_Z):
                yield

    for _ in move_to_ready():
        yield 

    # スクリーン座標からアームのXY座標を予測する。
    X = np.array(tcp_scrs)
    Y = np.array(arm_xyz)

    reg = LinearRegression().fit(X, Y)

    print('get_params', type(reg.get_params()), reg.get_params())
    print('coef_', type(reg.coef_), reg.coef_)
    print('intercept_', type(reg.intercept_), reg.intercept_)

    params['calibration']['hand-eye'] = {
        'coef': reg.coef_.tolist(), 
        'intercept': reg.intercept_.tolist()
    }

    params['X'] = X.tolist()
    params['Y'] = Y.tolist()

    write_params(params)

    prd = reg.predict(X)

    for dx, dy, dz in (Y - prd).tolist():
        print(f'dxyz 1:{dx:.1f} {dy:.1f} {dz:.1f}')

    A = (reg.coef_.dot(X.transpose()) + reg.intercept_.reshape(3, 1)).transpose()
    B = Y - A
    for i in range(X.shape[0]):
        dx, dy, dz = B[i, :]
        print(f'dxyz 2:{dx:.1f} {dy:.1f} {dz:.1f}')

    for (arm_x, arm_y, arm_z), height, (scr_x, scr_y) in zip(arm_xyz, tcp_heights, tcp_scrs):
        X = np.array([[scr_x, scr_y]])

        prd = reg.predict(X)
        prd_arm_x, prd_arm_y, prd_arm_z = prd[0, :]

        f.write(f'{scr_x}, {scr_y}, {height}, {arm_x}, {arm_y}, {arm_z}, {prd_arm_x}, {prd_arm_y}, {prd_arm_z}\n')

    f.close()




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


def show_next_pose(ch, servo_deg):
    degs = list(servo_angles)
    degs[ch] = servo_deg

    pose = forward_kinematics(degs)            
    show_pose(window, pose)

def get_arm_xyz_from_screen(scr_x, scr_y):
    coef = np.array(params['calibration']['hand-eye']['coef'])
    intercept = np.array(params['calibration']['hand-eye']['intercept'])

    arm_x, arm_y, arm_z = coef.dot(np.array([scr_x, scr_y])) + intercept

    return arm_x, arm_y, arm_z


def get_plane():
    p1, p2, p3 = [ Vec3(*xyz ) for xyz in marker_table[:3, :3].tolist() ]

    norm = (p2 - p1).cross(p3 - p1).unit()

    return norm, p1

def get_tcp():

    if np.isnan(marker_table[3:5, :]).any():
        return None, None, np.nan
    else:
        tcp_cam  = marker_table[3:5,  :3].mean(axis=0)
        tcp_scr  = marker_table[3:5, 3:5].mean(axis=0)

        tcp_cam = Vec3(* tcp_cam.tolist())
        tcp_scr = Vec2(* tcp_scr.tolist())

        if basis_point is None:
            tcp_height = np.nan

        else:
            tcp_height = abs(normal_vector.dot(tcp_cam - basis_point))

        return tcp_cam, tcp_scr, tcp_height

def prepare():
    global normal_vector, basis_point, camera_pred_x, camera_pred_y

    if np.isnan(marker_table[:, :3]).any():
        return False

    # 平面の法線ベクトルと、平面上の1点
    normal_vector, basis_point = get_plane()

    return True

def get_arm_xyz_of_work():
    if not be_prepared or inference is None:
        np.nan, np.nan

    work_scr_x, work_scr_y = inference.get(frame)
    if np.isnan(work_scr_x):
        return np.nan, np.nan

    arm_x, arm_y, arm_z = get_arm_xyz_from_screen(work_scr_x, work_scr_y)
    print(f'hand {arm_x:.1f} {arm_y:.1f} {arm_z:.1f}')

    return arm_x, arm_y, arm_z

def test_xyz():
    global test_pos

    h, w = frame.shape[:2]
    for scr_x in range(w // 3, w * 2 // 3, 70):
        for scr_y in range(h // 3, h * 2 // 3, 70):
            arm_x, arm_y, arm_z = get_arm_xyz_from_screen(scr_x, scr_y)

            for _ in move_xyz(arm_x, arm_y, LIFT_Z):
                yield

            for _ in move_xyz(arm_x, arm_y, arm_z):
                yield

            test_pos = Vec2(scr_x, scr_y)
            for _ in sleep(2):
                yield
            test_pos = None


def approach(arm_x, arm_y, arm_z):
    print('ready位置へ移動')
    for _ in move_to_ready():
        yield 

    print('ハンドを開く。')
    for _ in open_hand():
        yield

    print('ワークの把持のXY位置へ移動')
    for _ in move_xyz(arm_x, arm_y, LIFT_Z):
        yield

    print('把持位置を下げる。')
    for _ in move_xyz(arm_x, arm_y, arm_z):
        yield

    # for trial in range(10000):
    #     while np.isnan(tcp_height):
    #         yield

    #     diff = tcp_height - PICK_Z
    #     print(f'move z trial:{trial} diff:{diff:.1f}')
    #     if abs(diff) < 3:
    #         break

    #     # PICK_Zの位置に移動する。
    #     z -= tcp_height - PICK_Z
    #     for _ in move_xyz(x, y, z):
    #         yield





def grab(arm_x, arm_y, arm_z):
    for _ in approach(arm_x, arm_y, arm_z):
        yield

    while np.isnan(tcp_height):
        yield

    print('ハンドを閉じる。')
    for _ in close_hand():
        yield

    for _ in sleep(3):
        yield

    print('ワークを持ち上げる。')
    for _ in move_xyz(arm_x, arm_y, LIFT_Z):
        yield

    print('ready位置へ移動')
    for _ in move_to_ready():
        yield 

    print('ワークのプレース位置へ移動')
    for _ in move_linear(pose1):
        yield

    print('ハンドを開く。')
    for _ in open_hand():
        yield

    print('ready位置へ移動')
    for _ in move_to_ready():
        yield 



if __name__ == '__main__':
    normal_vector = None
    basis_point = None
    be_prepared = False

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

    marker_table = np.array([[0] * 6] * len(marker_ids), dtype=np.float32)
    layout = [
        [
            sg.Column([
                [
                    sg.Text('', key='-tcp-height-')
                ]
            ])
            ,
            sg.Column([
                spin2(f'J{ch+1}', f'J{ch+1}', deg, servo_to_angle(ch, deg), -120, 150, True) + [ 
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
            sg.Table(marker_table.tolist(), headings=['x', 'y', 'z', 'px', 'py', 'angle2'], auto_size_columns=False, col_widths=[6]*6, num_rows=len(marker_ids), key='-marker-table-')
        ]
        ,
        [ sg.Checkbox('grid', default=False, key='-show-grid-'), sg.Button('Ready'), sg.Button('Pose1'), sg.Button('test'), sg.Button('Prepare'), sg.Button('Calibrate'), sg.Button('Approach'), sg.Button('Grab'), sg.Button('Close')]
    ]

    window = sg.Window('calibration', layout, element_justification='c', finalize=True) # disable_minimize=True

    pose = forward_kinematics(servo_angles)
    show_pose(window, pose)

    last_capture = time.time()
    is_moving = False

    moving = None
    test_pos = None

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
            moving = move_to_ready()

        elif event == 'Pose1':
            moving = move_linear(pose1)

        elif event == 'test':
            moving = test_xyz()

        elif event == 'Approach':
            arm_x, arm_y, arm_z = get_arm_xyz_of_work()
            if not np.isnan(arm_x):

                moving = approach(arm_x, arm_y, arm_z)

        elif event == 'Grab':
            arm_x, arm_y, arm_z = get_arm_xyz_of_work()
            if not np.isnan(arm_x):

                moving = grab(arm_x, arm_y, arm_z)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            write_params(params)

            closeCamera()
            break

        elif event == 'Prepare':
            if not np.isnan(marker_table[:, :3]).any():
                prepare()

        elif event == "Calibrate":
            moving = calibrate_xy()        

        else:
            if 0.1 < time.time() - last_capture:
                last_capture = time.time()

                frame = getCameraFrame()

                if moving is None:
                    cx, cy = np.nan, np.nan
                    if inference is not None:
                        if moving is None:
                            cx, cy = inference.get(frame)
                        else:
                            cx, cy = inference.cx, inference.cy

                    if not np.isnan(cx):
                        if not be_prepared:
                            be_prepared = prepare()

                        if be_prepared:
                            arm_x, arm_y, arm_z = get_arm_xyz_from_screen(cx, cy)
                            print(f'screen {int(cx)} {int(cy)} hand:{arm_x:.1f} {arm_y:.1f} {arm_z:.1f}')

                tcp_height = np.nan
                if is_moving:

                    tcp_cam, tcp_scr = None, None
                else:
                    frame, vecs = detect_markers(marker_ids, frame)

                    for ch, vec in enumerate(vecs):                    
                        if vec is None:

                            marker_table[ch, :5] = [np.nan] * 5
                        else:

                            ivec = [int(x) for x in vec]

                            marker_table[ch, :5] = ivec[:5]

                            # zは常に正
                            marker_table[ch, 2] = np.abs(marker_table[ch, 2])


                    tcp_cam, tcp_scr, tcp_height = get_tcp()

                    window['-tcp-height-'].update(value=f'height:{tcp_height:.1f}')

                    window['-marker-table-'].update(values=marker_table.tolist())


                    if values['-show-grid-']:
                        draw_grid(frame)

                    if tcp_scr is not None:
                        cv2.circle(frame, (int(tcp_scr.x), int(tcp_scr.y)), 5, (0,255,0), -1)

                    if not np.isnan(cx):
                        cv2.circle(frame, (int(cx), int(cy)), 10, (255,0,0), -1)

                    if test_pos is not None:
                        cv2.circle(frame, (int(test_pos.x), int(test_pos.y)), 5, (0,0,255), -1)

                cv2.imshow("camera", frame)

                window.refresh()
