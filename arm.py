import time
import math
import numpy as np
import PySimpleGUI as sg
import json
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand
from kinematics import forward_kinematics, inverse_kinematics, move_linear, move_xyz
from util import nax, jKeys, radian, write_params, loadParams, t_all, spin, spin2, degree, Vec2, arctan2p, sleep, show_pose
from servo import init_servo, set_angle, move_joint, move_servo, servo_to_angle, angle_to_servo, move_all_joints
from infer import Inference
    
hand_idx = nax - 1
poseDim = 5
moving = None
stopMoving = False
dstAngles = [0] * nax
moveCnt = 30
grab_cnt = 0
Pos1 = [   0, -80, 80, 80,   0, 0  ]
Pos2 = [ -90, -80, 80, 80, -90, 0  ]

posKeys = [ "X", "Y", "Z", "R1", "R2" ]

def getPose():
    dst = [ float(values[k]) for k in posKeys ]
    dst[3] = radian(dst[3])
    dst[4] = radian(dst[4])

    return dst

def jointKey(i):
    return "J%d" % (i+1)

def showJoints(ts):
    for i, j in enumerate(ts):
        key = jointKey(i)
        window[key].Update(int(round(degree(j))))



def openHand():
    for _ in move_joint(hand_idx, -25):
        yield

def closeHand():
    for _ in move_joint(hand_idx, 20):
        yield

def grabWork(x, y):
    global grab_cnt

    grab_z = 10
    lift_z = 30

    for _ in move_linear(Pos1):
        yield

    for _ in openHand():
        yield

    for _ in move_xyz(x, y, grab_z):
        yield

    for _ in closeHand():
        yield

    for _ in sleep(3):
        yield

    x = (grab_cnt % 4) * 50
    grab_cnt += 1

    for _ in move_linear(Pos1):
        yield

    for _ in move_linear(Pos2):
        yield

    for _ in move_xyz(x, -150, lift_z):
        yield

    for _ in openHand():
        yield

    for _ in move_linear(Pos1):
        yield

if __name__ == '__main__':

    params, servo_angles, Angles = loadParams()

    init_servo(params, servo_angles, Angles)

    initCamera()

    inference = Inference()

    layout = [
        [
        sg.Column([
            spin2(f'J{i+1}', f'J{i+1}', servo_angles[i], degree(Angles[i]), -120, 120, True)
            for i in range(nax)
        ])
        ,
        sg.Column([
            spin('X', 'X' , 0,    0, 400 ),
            spin('Y', 'Y' , 0, -300, 300 ),
            spin('Z', 'Z' , 0,    0, 150 ),
            spin('R1', 'R1', 0, -90,  90 ),
            spin('R2', 'R2', 0,   0, 120 )
        ])
        ],
        [ sg.Button('Reset'), sg.Button('Ready'), sg.Button('Stop'), sg.Button('Send'), sg.Button('Calibrate'), sg.Button('Close')]
    ]

    # Create the Window
    window = sg.Window('Robot Control', layout, finalize=True)

    pose = forward_kinematics(servo_angles)
    show_pose(window, pose)

    servo_angle_keys = [ f'J{i+1}-servo' for i in range(nax) ]

    last_capture = time.time()
    while True:
        event, values = window.read(timeout=1)

        if moving is not None:
            try:
                moving.__next__()
            except StopIteration:
                moving = None
                stopMoving = False

                params['servo-angles'] = servo_angles
                write_params(params)

                pose = forward_kinematics(servo_angles)
                show_pose(window, pose)

                print("stop moving")

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
            
        elif event in posKeys:
            # 目標ポーズ

            pose = getPose()
            rad5s = inverse_kinematics(pose)
            if rad5s is not None:

                deg5s = degree(rad5s)

                for ch, deg in enumerate(deg5s):

                    window[jKeys[ch]].update(value=int(deg))
                    window[servo_angle_keys[ch]].update(value=int(angle_to_servo(ch, deg)))

                moving = move_linear(pose)

        elif event == "Stop":
            moving = None
            stopMoving = True
            
        elif event == "Reset":
            degs = [0] * nax
            degs[5] = degree(Angles[5])
            moving = move_all_joints(degs)
            
        elif event == "Ready":            
            moving = move_all_joints(Pos1)

        elif event == "Send":

            eye_x, eye_y = sendImage(values, inference)
            hand_x, hand_y = Eye2Hand(eye_x, eye_y)
            moving = grabWork(hand_x, hand_y)

        elif event == sg.WIN_CLOSED or event == 'Close':

            write_params(params)

            closeCamera()
            break

        else:
            if moving is None:
                if 0.1 < time.time() - last_capture:
                    last_capture = time.time()
                    readCamera(values)
        
    window.close()

