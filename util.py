import os
import sys
import math
import time
import json
import numpy as np
import PySimpleGUI as sg

nax = 6
move_time = 2

jKeys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

pose_keys = [ "X", "Y", "Z", "R1", "R2" ]

servo_angle_keys = [ f'J{i+1}-servo' for i in range(nax) ]

class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f'(x:{self.x:.1f} y:{self.y:.1f})'

    def len(self):
        return math.sqrt(self.x * self.x + self.y * self.y )

    def unit(self):
        l = self.len()

        assert(l != 0)

        return Vec2(self.x / l, self.y / l)

    def dot(self, v):
        return self.x * v.x + self.y * v.y

    def cross(self, v):
        return self.x * v.y - self.y * v.x

    def rot(self, t):
        cs = math.cos(t)
        sn = math.sin(t)

        return Vec2(self.x * cs - self.y * sn, self.x * sn + self.y * cs)

    def arctan2p(self):
        return arctan2p(self.y, self.x)

    def __rmul__(self, other):
        return Vec2(other * self.x , other * self.y)

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

class Vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f'(x:{self.x:.1f} y:{self.y:.1f} z:{self.z:.1f})'

    def len(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z )

    def unit(self):
        l = self.len()

        assert(l != 0)

        return Vec3(self.x / l, self.y / l, self.z / l)

    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z

    def cross(self, v):
        return Vec3(self.y * v.z - self.z * v.y, self.z * v.x - self.x * v.z, self.x * v.y - self.y * v.x)

    def __rmul__(self, other):
        return Vec3(other * self.x , other * self.y, other * self.z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

def get_move_time():
    return move_time

def set_move_time(t):
    global move_time

    old_move_time = move_time
    move_time = t

    return old_move_time


def sleep(sec):
    start_time = time.time()
    while time.time() - start_time < sec:
        yield

def radian(d):
    if type(d) is list:
        return [ x * math.pi / 180.0 for x in d ]
    else:
        return d * math.pi / 180.0

def degree(t):
    if type(t) is list:
        return [ x * 180.0 / math.pi for x in t ]
    else:
        return t * 180.0 / math.pi

def arctan2p(y, x):
    rad = np.arctan2(y, x)
    if 0 <= rad:
        return rad
    else:
        return 2 * np.pi + rad

def write_params(params):
    with open('data/arm.json', 'w') as f:
        json.dump(params, f, indent=4)

def read_params():
    if not os.path.isfile('data/arm.json'):

        params = {
            "COM": "COM?",
            "camera-index": 0,
            "prev-servo": [ 90 ] * 6,
            "servo-angle": [[ 1, 90]] * 6,
            "marker-ids": [ 0, 1, 2, 3, 4 ],
            "cameras" : {}
        }

        write_params(params)

        print('data/arm.json is created.')
        sys.exit(0)

    with open('data/arm.json', 'r') as f:
        params = json.load(f)

    return params

def spin(label, key, val, min_val, max_val, bind_return_key = True):

    return [ 
        sg.Text(label, size=(4, 1)),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(5, 1), key=key, enable_events=not bind_return_key, bind_return_key=bind_return_key )
    ]

def spin2(label, key, val1, val2, min_val, max_val, bind_return_key = True):

    return [ 
        sg.Text(label),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=int(val1), size=(5, 1), key=key+'-servo', enable_events=not bind_return_key, bind_return_key=bind_return_key ),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=int(val2), size=(5, 1), key=key, enable_events=not bind_return_key, bind_return_key=bind_return_key )
    ]


def get_pose(values):
    dst = [ float(values[k]) for k in pose_keys ]
    dst[3] = radian(dst[3])
    dst[4] = radian(dst[4])

    return dst

def show_pose(window, pose):
    for k, p in zip(["X", "Y", "Z"], pose[:3]):
        window[k].Update(int(round(p)))
        
    for k, p in zip(["R1", "R2"], pose[3:]):
        window[k].Update(int(round(degree(p))))
