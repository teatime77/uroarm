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

hand_idx = nax - 1

GRAB_Z = 5
LIFT_Z = 30

pose1 = [
      40,
    -110,
      50,
      math.pi / 2,
      math.pi / 2
]


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

def open_hand():
    for _ in move_joint(hand_idx, 50):
        yield

def close_hand():
    for _ in move_joint(hand_idx, 20):
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

def Calibrate():
    tcp_scrs = []
    arm_xy = []

    ys = [-60, 0, 60]
    xs = [120, 190, 250]
    # ys = [-60, 60]
    # xs = [150, 250]
    for y in ys:
        for x in xs:
            print(f'start move x:{x} y:{y}')

            # z = LIFT_Zの位置に移動する。
            z = LIFT_Z
            for _ in move_xyz(x, y, z):
                yield

            if not move_linear_ok:
                print(f'skip move x:{x} y:{y}')
                continue
            
            print("move xy end")
            for _ in sleep(3):
                yield

            while np.isnan(tcp_height):
                yield

            # GRAB_Zの位置に移動する。
            z -= tcp_height - GRAB_Z
            for _ in move_xyz(x, y, z):
                yield

            for _ in sleep(3):
                yield

            while np.isnan(tcp_height):
                yield

            print(f'move z end:{tcp_height:.1f}')

            while get_tcp()[0] is None:
                yield

            tcp_cam, tcp_scr = get_tcp()

            arm_xy.append([x, y])
            tcp_scrs.append([tcp_scr.x, tcp_scr.y])

            # z = LIFT_Zの位置に移動する。
            for _ in move_xyz(x, y, LIFT_Z):
                yield


    # スクリーン座標からアームのXY座標を予測する。
    X = np.array(tcp_scrs)
    Y = np.array(arm_xy)

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

    for dx, dy in (Y - prd).tolist():
        print(f'dxyz 1:{dx:.1f} {dy:.1f}')

    A = (reg.coef_.dot(X.transpose()) + reg.intercept_.reshape(2, 1)).transpose()
    B = Y - A
    for i in range(X.shape[0]):
        dx, dy = B[i, :]
        print(f'dxyz 2:{dx:.1f} {dy:.1f}')

def test_arm_from_screen():
    # スクリーン座標
    X = np.array(params['X'])

    # アームのXY座標の正解
    Y = np.array(params['Y'])

    coef = np.array(params['calibration']['hand-eye']['coef'])
    intercept = np.array(params['calibration']['hand-eye']['intercept'])

    # アームのXY座標の予測値
    Y2 = (coef.dot(X.transpose()) + intercept.reshape(2, 1)).transpose()

    # 予測の誤差
    D = Y - Y2
    for i in range(X.shape[0]):
        dx, dy = D[i, :]
        print(f'dxyz 2:{dx:.1f} {dy:.1f}')

        # スクリーン座標
        scr_x, scr_y = X[i, :]

        # アームのXY座標の予測値
        arm_x, arm_y = get_arm_xy_from_screen(scr_x, scr_y)

        # 予測の誤差
        dx, dy = Y[i, :] - np.array([arm_x, arm_y])
        print(f'dxyz 3:{dx:.1f} {dy:.1f}')


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

def show_next_pose(ch, servo_deg):
    degs = list(servo_angles)
    degs[ch] = servo_deg

    pose = forward_kinematics(degs)            
    show_pose(window, pose)

def get_camera_xz_yz_from_screen(screen_x, screen_y):
    # テスト用のマーカー
    X = np.array([screen_x, screen_y]).reshape((1, 2))

    # x/zとy/zを予測する。
    xz = camera_pred_x.predict(X)
    yz = camera_pred_y.predict(X)

    assert xz.shape == (1,) and yz.shape == (1,)
    xz, yz = xz[0], yz[0]

    return xz, yz

def get_camera_xyz_from_xz_yz(norm, p1, xz, yz):
    # 平面の方程式
    # n.x * x + n.y * y + n.z * z = n.dot(p1)

    # 両辺をzで割る。
    # n.x * x/z + n.y * y/z + n.z = n.dot(p1) / z

    # x/zとy/zからzを計算する。
    # z = n.dot(p1) / (n.x * x/z + n.y * y/z + n.z)
 
    # x/zとy/zからzを計算する。
    z = norm.dot(p1) / (norm.x * xz + norm.y * yz + norm.z)

    # x, y, zを計算する。
    x1, y1, z1 = xz * z, yz * z, z

    return x1, y1, z1

def get_arm_xy_from_screen(scr_x, scr_y):
    coef = np.array(params['calibration']['hand-eye']['coef'])
    intercept = np.array(params['calibration']['hand-eye']['intercept'])

    arm_x, arm_y = coef.dot(np.array([scr_x, scr_y])) + intercept

    return arm_x, arm_y

def make_screen_to_camera_predictor(norm, p1):
    # スクリーン座標
    X = marker_table[:, 3:5]

    # カメラ座標
    xz = marker_table[:, 0] / marker_table[:, 2]
    yz = marker_table[:, 1] / marker_table[:, 2]

    # スクリーン座標からカメラ座標のx/zとy/zの予測の学習をする。
    camera_pred_x = LinearRegression().fit(X, xz)
    camera_pred_y = LinearRegression().fit(X, yz)

    # 予測値を得る。
    prd_xz = camera_pred_x.predict(X)
    prd_yz = camera_pred_y.predict(X)

    # 予測の誤差を検証する。
    for i in range(X.shape[0]):
        diff_xz, diff_yz = prd_xz[i] - xz[i], prd_yz[i] - yz[i]
        if(0.005 < abs(diff_xz) or 0.005 < abs(diff_yz)):
            print(f'cam:{X[i, 0]} {X[i, 1]} x:({xz[i]:.3f} {prd_xz[i]:.3f}) y:({yz[i]:.3f} {prd_yz[i]:.3f}) diff:{diff_xz:.5f} {diff_yz:.5f}')

    return camera_pred_x, camera_pred_y

def get_plane():
    p1, p2, p3 = [ Vec3(*xyz ) for xyz in marker_table[:3, :3].tolist() ]

    norm = (p2 - p1).cross(p3 - p1).unit()

    return norm, p1

def get_tcp():

    if np.isnan(marker_table[3:5, :]).any():
        return None, None
    else:
        tcp_cam  = marker_table[3:5,  :3].mean(axis=0)
        tcp_scr  = marker_table[3:5, 3:5].mean(axis=0)

        return Vec3(* tcp_cam.tolist()), Vec2(* tcp_scr.tolist())

def prepare():
    global normal_vector, basis_point, camera_pred_x, camera_pred_y

    if np.isnan(marker_table[:, :3]).any():
        return False

    # 平面の法線ベクトルと、平面上の1点
    normal_vector, basis_point = get_plane()

    # スクリーン座標からカメラ座標のx/z, y/zへの学習器
    camera_pred_x, camera_pred_y = make_screen_to_camera_predictor(normal_vector, basis_point)

    # テスト用のポイントのスクリーン座標
    screen_x, screen_y = marker_table[5, 3:5]

    # カメラ座標のx/z, y/z
    xz, yz = get_camera_xz_yz_from_screen(screen_x, screen_y)

    # カメラ座標のx, y, z
    x1, y1, z1 = get_camera_xyz_from_xz_yz(normal_vector, basis_point, xz, yz)

    # 正解のx, y, z
    x2, y2, z2 = marker_table[5, :3]
    if 3 <= max(abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)):
        print(f'xyz:({x1:.1f} {x2:.1f}, {y1:.1f} {y2:.1f}, {z1:.1f} {z2:.1f})')

    return True

def get_arm_xy_of_work():
    if not be_prepared or inference is None:
        np.nan, np.nan

    work_scr_x, work_scr_y = inference.get(frame)
    if np.isnan(work_scr_x):
        return np.nan, np.nan

    arm_x, arm_y = get_arm_xy_from_screen(work_scr_x, work_scr_y)
    print(f'hand {arm_x:.1f} {arm_y:.1f}')

    return arm_x, arm_y

def approach(arm_x, arm_y):
    print('ready位置へ移動')
    for _ in move_all_joints(params['ready']):
        yield 

    print('ハンドを開く。')
    for _ in open_hand():
        yield

    print('ワークの把持のXY位置へ移動')
    for _ in move_xyz(arm_x, arm_y, LIFT_Z):
        yield

    print('把持位置を下げる。')
    for _ in move_xyz(arm_x, arm_y, GRAB_Z):
        yield


def grab(arm_x, arm_y):
    for _ in approach(arm_x, arm_y):
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
    for _ in move_all_joints(params['ready']):
        yield 

    print('ワークのプレース位置へ移動')
    for _ in move_linear(pose1):
        yield

    print('ハンドを開く。')
    for _ in open_hand():
        yield

    print('ready位置へ移動')
    for _ in move_all_joints(params['ready']):
        yield 

def approach_OLD(cx, cy):
    x1, y1, z1, _, _ = get_pose(values)

    while True:
        if(np.isnan(marker_table[3:, 3:5]).any()):
            yield
            continue

        p1 = marker_table[3, 3:5]
        p2 = marker_table[4, 3:5]

        pc = 0.5 * (p1 + p2)
        px, py = pc

        step = 5
        if px < cx:
            y1 += step
        else:
            y1 -= step

        if py < cy:
            x1 += step

        else:
            x1 -= step

        print(f'cx:{cx} cy:{cy} px:{px:.1f} py:{py:.1f} target:({x1:.1f}, {y1:.1f}, {z1:.1f})')

        for _ in move_xyz(x1, y1, z1):
            yield


if __name__ == '__main__':
    normal_vector = None
    basis_point = None
    be_prepared = False

    params = read_params()

    test_arm_from_screen()

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
        [ sg.Checkbox('grid', default=False, key='-show-grid-'), sg.Button('Ready'), sg.Button('Pose1'), sg.Button('Prepare'), sg.Button('Calibrate'), sg.Button('Approach'), sg.Button('Grab'), sg.Button('Close')]
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

        elif event == 'Pose1':
            show_pose(window, pose1)
            moving = move_linear(pose1)

        elif event == 'Approach':
            arm_x, arm_y = get_arm_xy_of_work()
            if not np.isnan(arm_x):

                moving = approach(arm_x, arm_y)

        elif event == 'Grab':
            arm_x, arm_y = get_arm_xy_of_work()
            if not np.isnan(arm_x):

                moving = grab(arm_x, arm_y)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            write_params(params)

            closeCamera()
            break

        elif event == 'Prepare':
            if not np.isnan(marker_table[:, :3]).any():
                prepare()

        elif event == "Calibrate":
            moving = Calibrate()        

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
                            arm_x, arm_y = get_arm_xy_from_screen(cx, cy)
                            print(f'screen {int(cx)} {int(cy)} hand:{arm_x:.1f} {arm_y:.1f} {GRAB_Z:.1f}')

                frame, vecs = detect_markers(marker_ids, frame)

                for ch, vec in enumerate(vecs):                    
                    if vec is None:

                        marker_table[ch, :5] = [np.nan] * 5
                    else:

                        ivec = [int(x) for x in vec]

                        marker_table[ch, :5] = ivec[:5]

                        # zは常に正
                        marker_table[ch, 2] = np.abs(marker_table[ch, 2])

                tcp_height = np.nan

                tcp_cam, tcp_scr = get_tcp()

                if all(x is not None for x in [tcp_cam, normal_vector, basis_point]):

                    tcp_height = abs(normal_vector.dot(tcp_cam - basis_point))

                window['-tcp-height-'].update(value=f'height:{tcp_height:.1f}')

                window['-marker-table-'].update(values=marker_table.tolist())


                if values['-show-grid-']:
                    draw_grid(frame)

                if tcp_scr is not None:
                    cv2.circle(frame, (int(tcp_scr.x), int(tcp_scr.y)), 5, (0,255,0), -1)

                if not np.isnan(cx):
                    cv2.circle(frame, (int(cx), int(cy)), 10, (255,0,0), -1)

                cv2.imshow("camera", frame)

                window.refresh()

                if is_first:

                    is_first = False
                
                    cv2.setMouseCallback('camera', mouse_callback)
