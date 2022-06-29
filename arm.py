import time
import math
import numpy as np
import PySimpleGUI as sg
import json
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand
from util import jKeys, radian, writeParams, loadParams, t_all, spin, spin2, degree, Vec2, arctan2p
from servo import init_servo, set_angle, move_joint, move_servo, servo_to_angle, angle_to_servo
from s_curve import SCurve
from infer import Inference

def arr(v):
    if isinstance(v, np.ndarray):
        return v
    else:
        return np.array(v, dtype=np.float32)
    
nax = 6
hand_idx = nax - 1
poseDim = 5
moving = None
stopMoving = False
dstAngles = [0] * nax
moveCnt = 30
grab_cnt = 0
Pos1 = [   0, -80, 80, 80,   0, 0  ]
Pos2 = [ -90, -80, 80, 80, -90, 0  ]

zs = arr([ 0.5, 11.0, 17.5, 23.5, 28.8 ]) - 0.5
links = [ zs[i+1] - zs[i] for i in range(4) ]

L0, L1, L2, L3, L4 = [ 111, 105, 98, 25, 162 ]

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


def moveAllJoints(dsts):
    srcs = [degree(x) for x in Angles]

    scs = [ SCurve(dsts[j] - srcs[j]) for j in range(nax) ]

    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t_all <= t:
            break

        for j in range(nax):
            sc = scs[j]
            
            deg = srcs[j] + sc.dist(t)
            set_angle(j, deg)

        yield

    print("move end %d msec" % int(1000 * (time.time() - start_time) / moveCnt))

def waitMoveAllJoints(dst):
    mv = moveAllJoints(dst)
    while True:                
        try:
            mv.__next__()
            yield True
        except StopIteration:
            yield False

def waitMoveJoint(jkey, dst_deg):
    jidx = jKeys.index(jkey)
    src_deg = degree(Angles[jidx])

    sc = SCurve(dst_deg - src_deg)

    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t_all <= t:
            yield False
            
        deg = src_deg + sc.dist(t)
        set_angle(jidx, deg)

        yield True

def waitPos(pos):
    degs = list(pos)
    degs[hand_idx] = degree(Angles[hand_idx])

    mv = waitMoveAllJoints(degs)
    while mv.__next__():
        yield True

    yield False

def move_linear(dst):
    src = calc(Angles)

    with open('ik.csv', 'w') as f:
        f.write('time,J1,J2,J3,J4,J5,J6\n')
        start_time = time.time()
        while True:
            t = time.time() - start_time
            if t_all <= t:
                break

            r = t / t_all

            pose = [ r * d + (1 - r) * s for s, d in zip(src, dst) ]

            rads = IK(pose)
            if rads is not None:
                degs = degree(rads)
                f.write(f'{t},{",".join(["%.1f" % x for x in degs])}\n')

                for ch, deg in enumerate(degs):
                    set_angle(ch, deg)

            yield

    print("move end %d msec" % int(1000 * (time.time() - start_time) / moveCnt))


def calc(rads):
    j0, j1, j2, j3, j4, j5 = rads

    p1 = Vec2(0, L0)

    p2 = p1 + (Vec2(-L1,0).rot(j1))
    print(f'p2: {int(p2.x), int(p2.y)}')

    p3 = p2 + L2 * (p2 - p1).unit().rot(j2)
    print(f'p3: %d %d' % (int(p3.x), int(p3.y)))

    p4 = p3 + L3 * (p3 - p2).unit().rot(j3 - 0.5 * np.pi)
    print(f'p4: %d %d' % (int(p4.x), int(p4.y)))

    tcp = p4 + L4 * (p3 - p2).unit().rot(j3)
    print(f'tcp: %d %d' % (int(tcp.x), int(tcp.y)))

    et = (p4 - tcp).unit()
    theta = np.arctan2(et.y, et.x)

    r = tcp.len()

    x = r * math.cos(j0)
    y = r * math.sin(j0)
    z = tcp.y

    # print('r:%.1f j0:%.1f x:%.1f y:%.1f z:%.1f' % (r, degree(j0), x, y, z))

    return arr([ x, y, z, j4, theta])



def IK(pose):
    x, y, z, phi, theta = pose

    r   = Vec2(x, y).len()

    p1 = Vec2(0, L0)

    tcp = Vec2(-r, z)
    p4 = tcp + L4 * Vec2(math.cos(theta), math.sin(theta))
    # print(f'p4: %d %d' % (int(- p4.x), int(p4.y)))

    # tcpからp4に向かう単位ベクトル
    et = (p4 - tcp).unit()

    # p4からp3に向かう単位ベクトル
    e4 = et.rot(radian(-90))

    p3 = p4 + L3 * e4
    # print(f'p3: %d %d' % (int(- p3.x), int(p3.y)))

    l = (p3 - p1).len()

    cos_alpha = (L2 * L2 + l * l - L1 * L1) / (2 * L2 * l)
    if 1 < abs(cos_alpha):
        print("cos alpha:%.2f l:%.1f  L1:%.1f  L2:%.1f" % (cos_alpha, l, L1, L2))
        return None

    alpha = math.acos(cos_alpha)    

    # p3からp1に向かう単位ベクトル
    e31 = (p1 - p3).unit()

    # p3からp2に向かう単位ベクトル
    e32 = e31.rot(alpha)

    # print('e31:%.2f %.2f e32:%.2f %.2f alpha:%.1f' % (e31.x, e31.y, e32.x, e32.y, degree(alpha)))

    p2 = p3 + L2 * e32
    # print(f'p2: %d %d' % (int(- p2.x), int(p2.y)))

    ts = [0] * nax

    ts[0] = np.arctan2(y, x)

    ts[1] = (p2 - p1).arctan2p() - np.pi

    ts[2] = (p3 - p2).arctan2p() - (p2 - p1).arctan2p()

    ts[3] = (tcp - p4).arctan2p() - (p3 - p2).arctan2p()

    ts[4] = phi

    ts[5] = Angles[5]

    # print("J2:%.1f  J3:%.1f  J4:%.1f " % (degree(ts[1]), degree(ts[2]), degree(ts[3])))

    return ts

def showPos(pos):
    for k, p in zip(["X", "Y", "Z"], pos[:3]):
        window[k].Update(int(round(p)))
        
    for k, p in zip(["R1", "R2"], pos[3:]):
        window[k].Update(int(round(degree(p))))


def openHand():
    mv = waitMoveJoint('J6', -25)
    while mv.__next__():
        yield True

    yield False

def closeHand():
    mv = waitMoveJoint('J6', 20)
    while mv.__next__():
        yield True

    yield False

def grabWork(x, y):
    global grab_cnt

    x += 10

    mv = waitPos(Pos1)
    while mv.__next__():
        yield True

    mv = openHand()
    while mv.__next__():
        yield True

    dst_rad = IK2(x, y, True)

    if dst_rad is None:
        print(f'skip move x:{x} y:{y}')
        return
    
    dst_deg = [degree(rad) for rad in dst_rad]

    mv = waitMoveAllJoints(dst_deg)
    while mv.__next__():
        yield

    mv = closeHand()
    while mv.__next__():
        yield True
    
    start_time = time.time()
    while time.time() - start_time < 3:
        yield

    x = (grab_cnt % 4) * 50
    grab_cnt += 1

    mv = waitPos(Pos1)
    while mv.__next__():
        yield True

    mv = waitPos(Pos2)
    while mv.__next__():
        yield True

    mv = waitMoveXY(x, -150)
    while mv.__next__():
        yield True

    mv = openHand()
    while mv.__next__():
        yield True

    mv = waitPos(Pos1)
    while mv.__next__():
        yield True

if __name__ == '__main__':

    params, servo_angles, Angles = loadParams()

    init_servo(params, servo_angles, Angles)

    initCamera()

    inference = Inference()

    layout = [
        [
        sg.Column([
            spin2(f'J{i+1}', f'J{i+1}', servo_angles[i], degree(Angles[i]), -120, 120, True)
            for i in range(6)
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

    pos = calc(Angles)
    showPos(pos)

    servo_angle_keys = [ f'J{i+1}-servo' for i in range(6) ]

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
                writeParams(params)

                pos = calc(Angles)
                showPos(pos)

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
            rads = IK(pose)
            if rads is not None:

                degs = degree(rads)

                for ch, deg in enumerate(degs):

                    window[jKeys[ch]].update(value=int(deg))
                    window[servo_angle_keys[ch]].update(value=int(angle_to_servo(ch, deg)))

                moving = move_linear(pose)

        elif event == "Stop":
            moving = None
            stopMoving = True
            
        elif event == "Reset":
            degs = [0] * nax
            degs[5] = degree(Angles[5])
            moving = moveAllJoints(degs)
            
        elif event == "Ready":            
            moving = moveAllJoints(Pos1)

        elif event == "Send":

            eye_x, eye_y = sendImage(values, inference)
            hand_x, hand_y = Eye2Hand(eye_x, eye_y)
            moving = grabWork(hand_x, hand_y)

        elif event == sg.WIN_CLOSED or event == 'Close':

            writeParams(params)

            closeCamera()
            break

        else:
            if moving is None:
                if 0.1 < time.time() - last_capture:
                    last_capture = time.time()
                    readCamera(values)
        
    window.close()

