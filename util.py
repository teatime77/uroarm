import math
import json
import PySimpleGUI as sg

t_all = 2

jKeys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

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

def writeParams(params):
    with open('data/arm.json', 'w') as f:
        json.dump(params, f, indent=4)

def loadParams():
    with open('data/arm.json', 'r') as f:
        params = json.load(f)

    com_port = params['COM'] 

    offsets = params['offsets']

    for key in jKeys:
        offsets[key] = float(offsets[key])

    scales  = [ float(x) for x in params['scales'] ]

    degrees = [ float(s) for s in params['degrees'] ]

    Angles  = radian(degrees)

    return params, com_port, offsets, scales, Angles

def spin(label, key, val, min_val, max_val, bind_return_key = True):

    return [ 
        sg.Text(label),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(5, 1), key=key, enable_events=not bind_return_key, bind_return_key=bind_return_key )
    ]
