import time
import math
import numpy as np
import PySimpleGUI as sg
import json
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand
from util import nax, jKeys, radian, write_params, t_all, spin, spin2, degree, Vec2, arctan2p
from servo import servo_to_angle

L0, L1, L2, L3, L4 = [ 111, 105, 98, 25, 162 ]

def arr(v):
    if isinstance(v, np.ndarray):
        return v
    else:
        return np.array(v, dtype=np.float32)

def forward_kinematics(servo_angles):
    rads = [radian(servo_to_angle(ch, deg)) for ch, deg in enumerate(servo_angles) ]

    j0, j1, j2, j3, j4, j5 = rads

    p1 = Vec2(0, L0)

    p2 = p1 + (Vec2(L1,0).rot(0.5 * np.pi + j1))

    p3 = p2 + L2 * (p2 - p1).unit().rot(j2)

    p4 = p3 + L3 * (p3 - p2).unit().rot(j3 + 0.5 * np.pi)

    tcp = p4 + L4 * (p3 - p2).unit().rot(j3)

    et = (p4 - tcp).unit()
    theta = normalize_radian(np.pi - np.arctan2(et.y, et.x))

    r = abs(tcp.x)

    x = r * math.cos(j0)
    y = r * math.sin(j0)
    z = tcp.y

    # print('r:%.1f j0:%.1f x:%.1f y:%.1f z:%.1f' % (r, degree(j0), x, y, z))

    pose = [ x, y, z, j4, theta]

    r   = Vec2(x, y).len()
    # print(f'FK p1:{p1} p2:{p2} p3:{p3} p4:{p4} tcp:{tcp} x:{x:.1f} y:{y:.1f} z:{z:.1f} theta:{degree(theta):.1f} r:{r:.1f}')
    # print('   js', [f'{degree(j):.1f}' for j in rads])


    rad5s = inverse_kinematics(pose)
    if rad5s is None:
        return arr(pose)

    degs1 = degree(np.array(rads[:5]))
    degs2 = degree(rad5s)
    
    diff_max = np.array([abs(deg2 - deg1) for deg1, deg2 in zip(degs1, degs2)]).max()
    if 0.1 < diff_max:
        print(f'IK diff:{diff_max:.16f}', [ f'{deg1:.1f} {deg2:.1f} {deg2 - deg1:.1f}'  for i, (deg1, deg2) in enumerate(zip(degs1, degs2)) ])

    return arr(pose)

def normalize_radian(rad):
    while np.pi < rad:
        rad -= 2 * np.pi

    while rad < -np.pi:
        rad += 2 * np.pi

    return rad

def inverse_kinematics(pose):
    x, y, z, phi, theta = pose

    r   = Vec2(x, y).len()

    if x < 0:
        r *= -1

    p1 = Vec2(0, L0)

    tcp = Vec2(r, z)

    theta2 = np.pi - theta
    p4 = tcp + L4 * Vec2(math.cos(theta2), math.sin(theta2))
    # print(f'p4: %d %d' % (int(- p4.x), int(p4.y)))

    # tcpからp4に向かう単位ベクトル
    et = (p4 - tcp).unit()

    # p4からp3に向かう単位ベクトル
    e4 = et.rot(radian(90))

    p3 = p4 + L3 * e4
    # print(f'p3: %d %d' % (int(- p3.x), int(p3.y)))

    l = (p3 - p1).len()

    cos_alpha = (L2 * L2 + l * l - L1 * L1) / (2 * L2 * l)
    if 1 < abs(cos_alpha):
        if 0.05 < abs(cos_alpha) - 1:
            print(f'cos alpha:{cos_alpha:.16f} l:{l:.1f}  L1:{L1:.1f}  L2:{L2:.1f} p1:{p1} p3:{p3} p4:{p4} tcp:{tcp} x:{x:.1f} y:{y:.1f} z:{z:.1f} r:{r:.1f}')
            return None
        else:
            print(f'cos_alpha:{cos_alpha:.16f} => {np.sign(cos_alpha):.16f}')
            cos_alpha = np.sign(cos_alpha)

    alpha = math.acos(cos_alpha)    

    # p3からp1に向かう単位ベクトル
    e31 = (p1 - p3).unit()

    # p3からp2に向かう単位ベクトル
    e32 = e31.rot(-alpha)

    # print('e31:%.2f %.2f e32:%.2f %.2f alpha:%.1f' % (e31.x, e31.y, e32.x, e32.y, degree(alpha)))

    p2 = p3 + L2 * e32
    # print(f'p2: %d %d' % (int(- p2.x), int(p2.y)))

    ts = [0] * (nax - 1)

    ts[0] = np.arctan2(y, x)

    ts[1] = (p2 - p1).arctan2p() - 0.5 * np.pi

    ts[2] = (p3 - p2).arctan2p() - (p2 - p1).arctan2p()

    ts[3] = (tcp - p4).arctan2p() - (p3 - p2).arctan2p()

    ts[4] = phi

    ts = [ normalize_radian(rad) for rad in ts ]

    # print(f'IK p1:{p1} p2:{p2} p3:{p3} p4:{p4} tcp:{tcp} x:{x:.1f} y:{y:.1f} z:{z:.1f} theta:{degree(theta):.1f} r:{r:.1f}')
    # print('   js', [f'{degree(j):.1f}' for j in ts])

    return ts
