import math
import json
import numpy as np
import PySimpleGUI as sg

t_all = 2

jKeys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']


class Vec2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

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

class Glb:
    def __init__(self):
        self.regX = None
        self.regY = None
        self.inferX = None
        self.inferY = None
        self.prdX = None
        self.prdY = None

theGlb = Glb()

def getGlb():
    return theGlb

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

def writeParams(params):
    with open('data/arm.json', 'w') as f:
        json.dump(params, f, indent=4)

def loadParams():
    with open('data/arm.json', 'r') as f:
        params = json.load(f)

    com_port = params['COM'] 

    offsets = params['offsets']

    scales  = [ float(x) for x in params['scales'] ]

    degrees = [ float(s) for s in params['degrees'] ]

    Angles  = radian(degrees)

    return params, com_port, offsets, scales, degrees, Angles

def spin(label, key, val, min_val, max_val, bind_return_key = True):

    return [ 
        sg.Text(label),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(5, 1), key=key, enable_events=not bind_return_key, bind_return_key=bind_return_key )
    ]
