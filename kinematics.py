import time
import math
import numpy as np
import PySimpleGUI as sg
import json
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand
from util import nax, jKeys, radian, write_params, loadParams, t_all, spin, spin2, degree, Vec2, arctan2p

L0, L1, L2, L3, L4 = [ 111, 105, 98, 25, 162 ]

def arr(v):
    if isinstance(v, np.ndarray):
        return v
    else:
        return np.array(v, dtype=np.float32)

def forward_kinematics(rads):
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


def inverse_kinematics(pose):
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
