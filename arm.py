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

PICK_Z = 15
CALIBRATION_Z = PICK_Z + 10
LIFT_Z = PICK_Z + 20
LIFT_Z2 = 60
PLACE_Z = PICK_Z + 10

pose1 = [ 90, -90, LIFT_Z2, math.pi / 4, math.pi / 2 ]
pose2 = [ 90, -90, PLACE_Z, math.pi / 4, math.pi / 2 ]


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
    for _ in move_joint(hand_idx, 100):
        yield

def close_hand():
    for _ in move_joint(hand_idx, 30):
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

def get_pose_from_xyz(x, y, z):
    x_min = 100
    x_max = 300

    pitch_min = 90
    pitch_max = 60

    assert x_min <= x and x <= x_max, f'get pose from xyz:{x_min:.1f} {x:.1f} {x_max:.1f}'

    r = (x - x_min) / (x_max - x_min)

    pitch = r * pitch_max + (1 - r) * pitch_min

    yaw = - math.atan2(y, x)
    pose = [ x, y, z, yaw, radian(pitch)]

    return pose

def move_xyz_now(x, y, z):
    pose = get_pose_from_xyz(x, y, z)
    rad5s = inverse_kinematics(pose)

    if rad5s is not None:
        deg5s = degree(rad5s) 

        for ch, deg in enumerate(deg5s):
            set_angle(ch, deg)


def move_xyz(x, y, z):
    pose = get_pose_from_xyz(x, y, z)
    for _ in move_linear(pose):
        yield

def move_to_ready():
    degs = [ 0, 0, -90, -90, 0, 0 ]

    for _ in move_all_joints(degs):
        yield


def calibrate_xy():
    global tcp_height

    tcp_scrs = []
    arm_xyz = []
    tcp_heights = []
    num_trial = 2
    num_points = 3

    for _ in range(num_trial):
        for arm_y in np.linspace(-60, 60, num_points):
            for arm_x in np.linspace(120, 230, num_points):
                print(f'start move x:{arm_x} y:{arm_y}')

                # z = LIFT_Zの位置に移動する。
                arm_z = LIFT_Z
                for _ in move_xyz(arm_x, arm_y, arm_z):
                    yield
                
                print("move xy end")
                for _ in sleep(3):
                    yield
                tcp_height = np.nan

                for trial in range(10000):
                    while np.isnan(tcp_height):
                        yield

                    diff = tcp_height - CALIBRATION_Z
                    print(f'move z trial:{trial} height:{tcp_height:.1f}')
                    if abs(diff) < 2:
                        break

                    # PICK_Zの位置に移動する。
                    arm_z -= diff
                    for _ in move_xyz(arm_x, arm_y, arm_z):
                        yield

                    for _ in sleep(1):
                        yield

                    tcp_height = np.nan

                print(f'move z end:{tcp_height:.1f}')

                tcp_heights.append(tcp_height)
                arm_xyz.append([arm_x, arm_y, arm_z])
                tcp_scrs.append([tcp_scr.x, tcp_scr.y])

                # z = LIFT_Zの位置に移動する。
                for _ in move_xyz(arm_x, arm_y, 50):
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

    params['hand-eye'] = {
        'coef': reg.coef_.tolist(), 
        'intercept': reg.intercept_.tolist()
    }

    write_params(params)

    prd = reg.predict(X)

    for dx, dy, dz in (Y - prd).tolist():
        print(f'dxyz 1:{dx:.1f} {dy:.1f} {dz:.1f}')

    A = (reg.coef_.dot(X.transpose()) + reg.intercept_.reshape(3, 1)).transpose()
    B = Y - A
    for i in range(X.shape[0]):
        dx, dy, dz = B[i, :]
        print(f'dxyz 2:{dx:.1f} {dy:.1f} {dz:.1f}')

    with open('data/calibrate-xy.csv', 'w') as f:

        f.write('scr-x,scr-y,tcp-height,arm-x,arm-y,arm-z,prd-arm-x,prd-arm-y,prd-arm-z\n')

        for (arm_x, arm_y, arm_z), height, (scr_x, scr_y) in zip(arm_xyz, tcp_heights, tcp_scrs):
            X = np.array([[scr_x, scr_y]])

            prd = reg.predict(X)
            prd_arm_x, prd_arm_y, prd_arm_z = prd[0, :]

            f.write(f'{scr_x}, {scr_y}, {height}, {arm_x}, {arm_y}, {arm_z}, {prd_arm_x}, {prd_arm_y}, {prd_arm_z}\n')



def show_next_pose(ch, servo_deg):
    degs = list(servo_angles)
    degs[ch] = servo_deg

    pose = forward_kinematics(degs)            
    show_pose(window, pose)

def get_arm_xyz_from_screen(scr_x, scr_y):
    if not 'hand-eye' in params:
        print('No hand-eye calibration')
        sys.exit(0)

    coef = np.array(params['hand-eye']['coef'])
    intercept = np.array(params['hand-eye']['intercept'])

    arm_x, arm_y, arm_z = coef.dot(np.array([scr_x, scr_y])) + intercept

    return arm_x, arm_y, arm_z

def get_tcp():

    if np.isnan(marker_table[3:, :]).any():
        return None, None, np.nan
    else:
        tcp_cam  = marker_table[3:,  :3].mean(axis=0)
        tcp_scr  = marker_table[3:, 3:5].mean(axis=0)

        tcp_cam = Vec3(* tcp_cam.tolist())
        tcp_scr = Vec2(* tcp_scr.tolist())

        if basis_point is None:
            tcp_height = np.nan

        else:
            tcp_height = abs(normal_vector.dot(tcp_cam - basis_point))

        return tcp_cam, tcp_scr, tcp_height

def set_plane(vecs):
    if any(vec is None for vec in vecs):
        plane_points.clear()
        return None, None

    plane_points.append(vecs)

    if len(plane_points) < 10:
        return None, None

    points_samples = np.array(plane_points)
    assert points_samples.shape == (10, 3, 5)

    points = points_samples.mean(axis=0)
    assert points.shape == (3, 5)

    p1, p2, p3 = [ Vec3(*xyz ) for xyz in points[:, :3].tolist() ]

    normal_vector = (p2 - p1).cross(p3 - p1).unit()

    basis_point = (1.0 / 3.0) * (p1 + p2 + p3)

    return normal_vector, basis_point

def get_arm_xyz_of_work():
    if inference is None:
        return [np.nan] * 5

    work_scr_x, work_scr_y = inference.get(frame)
    if np.isnan(work_scr_x):
        return [np.nan] * 5

    arm_x, arm_y, arm_z = get_arm_xyz_from_screen(work_scr_x, work_scr_y)
    print(f'hand {arm_x:.1f} {arm_y:.1f} {arm_z:.1f}')

    return work_scr_x, work_scr_y, arm_x, arm_y, arm_z

def test_xyz():
    global test_pos

    h, w = frame.shape[:2]
    for scr_x in np.linspace(w / 3, w * 2 / 3, 4):
        for scr_y in np.linspace(h // 3, h * 2 // 3, 4):
            arm_x, arm_y, arm_z = get_arm_xyz_from_screen(scr_x, scr_y)

            for _ in move_xyz(arm_x, arm_y, LIFT_Z):
                yield

            for _ in move_xyz(arm_x, arm_y, arm_z):
                yield

            test_pos = Vec2(scr_x, scr_y)
            for _ in sleep(1):
                yield
            test_pos = None

    for _ in move_to_ready():
        yield


def grab(work_scr_x, work_scr_y, arm_x, arm_y, arm_z):
    print('ready位置へ移動')
    for _ in move_to_ready():
        yield 

    print('ハンドを開く。')
    for _ in open_hand():
        yield

    print('ワークの把持のXY位置へ移動')
    for _ in move_xyz(arm_x, arm_y, LIFT_Z2):
        yield

    print('把持位置を下げる。')
    z = arm_z - (CALIBRATION_Z - PICK_Z)
    for _ in move_xyz(arm_x, arm_y, z):
        yield

    for _ in sleep(1):
        yield

    print('ハンドを閉じる。')
    for _ in close_hand():
        yield

    for _ in sleep(1):
        yield

    print('ワークを持ち上げる。')
    for _ in move_xyz(arm_x, arm_y, LIFT_Z2):
        yield


    print('ワークのリフト位置へ移動')
    for _ in move_linear(pose1):
        yield

    print('ワークのプレース位置へ移動')
    for _ in move_linear(pose2):
        yield

    print('ハンドを開く。')
    for _ in open_hand():
        yield

    print('ワークのリフト位置へ移動')
    for _ in move_linear(pose1):
        yield

    print('ready位置へ移動')
    for _ in move_to_ready():
        yield 



if __name__ == '__main__':
    plane_points = []
    normal_vector = None
    basis_point = None

    params = read_params()

    marker_ids = params['marker-ids']

    init_servo(params)
    init_markers(params)
    initCamera(params)


    if len(sys.argv) == 2:
        from infer import Inference

        inference = Inference()

    else:
        inference = None

    marker_table = np.array([[0] * 5] * len(marker_ids), dtype=np.float32)
    layout = [
        [
            sg.Column([
                spin('X', 'X' , 0,    0, 400 ),
                spin('Y', 'Y' , 0, -300, 300 ),
                spin('Z', 'Z' , 0,    0, 150 ),
                spin('R1', 'R1', 0, -90,  90 ),
                spin('R2', 'R2', 0,   0, 120 ),
                spin('hand', 'hand', 0,   0, 100 )
            ])
            ,
            sg.Column([
                [
                    sg.Table(marker_table.tolist(), headings=['cam x', 'cam y', 'cam z', 'scr x', 'scr y'], auto_size_columns=False, col_widths=[6]*5, num_rows=len(marker_ids), key='-marker-table-')
                ]
                ,
                [
                    sg.Text('', key='-tcp-height-')
                ]
            ])
        ]
        ,
        [ sg.Button('Reset'), sg.Button('Ready'), sg.Button('Pose1'), sg.Button('test'), sg.Button('Calibrate'), sg.Button('Grab'), sg.Button('Close')]
    ]

    window = sg.Window('Robot Arm', layout, element_justification='c', finalize=True) # disable_minimize=True

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

                params['prev-servo'] = servo_angles
                write_params(params)

        if event in pose_keys:
            pose = get_pose(values)
            moving = move_linear(pose)

        elif event == 'hand':

            deg = float(values[event])
            moving = move_joint(hand_idx, deg)            

        elif event == 'Ready':
            moving = move_to_ready()

        elif event == 'Pose1':
            moving = move_linear(pose1)

        elif event == 'test':
            moving = test_xyz()

        elif event == 'Grab':
            work_scr_x, work_scr_y, arm_x, arm_y, arm_z = get_arm_xyz_of_work()
            if not np.isnan(arm_x):

                moving = grab(work_scr_x, work_scr_y, arm_x, arm_y, arm_z)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['prev-servo'] = servo_angles
            write_params(params)

            closeCamera()
            if inference is not None:
                inference.close()

            break

        elif event == "Calibrate":
            moving = calibrate_xy()   

        elif event == 'Reset':     
            normal_vector = None
            basis_point = None
            plane_points.clear()

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

                tcp_height = np.nan
                if is_moving:

                    tcp_cam, tcp_scr = None, None
                else:
                    frame, vecs = detect_markers(marker_ids, frame)

                    for ch, vec in enumerate(vecs):                    
                        if vec is None:

                            marker_table[ch, :] = [np.nan] * 5
                        else:

                            marker_table[ch, :] = vec

                            # zは常に正
                            marker_table[ch, 2] = np.abs(marker_table[ch, 2])

                    if normal_vector is None:
                     
                        normal_vector, basis_point = set_plane(vecs[:3])


                    tcp_cam, tcp_scr, tcp_height = get_tcp()

                    window['-tcp-height-'].update(value=f'height:{tcp_height:.1f}')

                    window['-marker-table-'].update(values=np.round(marker_table).tolist())

                    if tcp_scr is not None:
                        cv2.circle(frame, (int(tcp_scr.x), int(tcp_scr.y)), 5, (0,255,0), -1)

                    if not np.isnan(cx):
                        cv2.circle(frame, (int(cx), int(cy)), 10, (255,0,0), -1)

                    if test_pos is not None:
                        cv2.circle(frame, (int(test_pos.x), int(test_pos.y)), 5, (0,0,255), -1)

                h, w = frame.shape[:2]
                assert h == w
                if h <= 720:
                    frame2 = frame
                else:
                    frame2 = cv2.resize(frame, (720, 720))

                cv2.imshow("camera", frame2)

                window.refresh()
