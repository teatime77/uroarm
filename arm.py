import time
import math
import numpy as np
import PySimpleGUI as sg
import json

try:
    import Adafruit_PCA9685

    useSerial = False

except ModuleNotFoundError:
    print("no Adafruit")

    import serial

    useSerial = True


def arr(v):
    if isinstance(v, np.ndarray):
        return v
    else:
        return np.array(v, dtype=np.float32)
    
nax = 6
moving = None
stopMoving = False
Angles = [0] * nax
dstAngles = [0] * nax

zs = arr([ 0.5, 11.0, 17.5, 23.5, 28.8 ]) - 0.5
links = [ zs[i+1] - zs[i] for i in range(4) ]

L0, L1, L2, L3, L4 = [ 111, 105, 98, 25, 162 ]

jKeys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
posKeys = [ "X", "Y", "Z", "R1", "R2" ]


def degree(t):
    return t * 180.0 / math.pi

def radian(d):
    return d * math.pi / 180.0

def loadParams():
    global offsets

    with open('arm.json', 'r') as f:
        params = json.load(f)

    offsets = params['offsets']

    for key in jKeys:
        offsets[key] = float(offsets[key])

    print("offsets", offsets)


def saveParams():
    obj = {}
    for key in jKeys:
        obj[key] = "%.1f" % offsets[key]


    params = {
        'offsets' : obj
    }

    with open('arm.json', 'w') as f:
        json.dump(params, f)

    print("saved:" + json.dumps(params))

def getPos():
    dst = [ values[k] for k in posKeys ]
    dst[3] = radian(dst[3])
    dst[4] = radian(dst[4])

    return dst

def arctan2p(y, x):
    rad = np.arctan2(y, x)
    if 0 <= rad:
        return rad
    else:
        return 2 * np.pi + rad

def strPos(pos):
    x, y, z, r1, r2 = pos
    return "x:%.1f, y:%.1f, z:%.1f, r1:%.1f, r2:%.1f" % (x, y, z, degree(r1), degree(r2))
    
def setPWM(ch, pos):
    pulse = round( 150 + (600 - 150) * (pos + 0.5 * math.pi) / math.pi )
    pwm.set_pwm(ch, 0, pulse)

def setAngleNano(event, t):
    m = { "J1":0, "J2":1, "J3":2, "J4":3, "J5":4, "J6":5 }

    v = [ 90, 95, 103, 105, 90, 90 ]

    i = jKeys.index(event)
    t *= float(v[i]) / 90.0

    t += radian(offsets[event])

    if event == "J2":
        t *= -1

    ch = m[event]
    # print("move %s ch:%d pos:%.1f" % (event, ch, t))
    setPWM(ch, t)


def setAngle(event, t):
    deg = degree(t)
    ch = jKeys.index(event)

    deg += offsets[event]

    if event == "J2":
        deg *= -1

    v = [ 90, 97, 105, 100, 90, 90 ]

    deg *= float(v[ch]) / 90.0

    deg += 90
    print(event, ch, deg, t)

    cmd = "%d,%.1f\r" % (ch, deg)
    n = ser.write(cmd.encode('utf-8'))

    ret = ser.readline().decode('utf-8')
    print("read", ret.strip())

    return

    m = { "J1":0, "J2":1, "J3":2, "J4":3, "J5":4, "J6":5 }

    ch = m[event]
    # print("move %s ch:%d pos:%.1f" % (event, ch, t))
    setPWM(ch, t)

def getAngles():
    return [radian(values[x]) for x in jKeys]

def jointKey(i):
    return "J%d" % (i+1)

def showJoints(ts):
    for i, j in enumerate(ts):
        key = jointKey(i)
        window[key].Update(degree(j))

def setOffsets():
    for key in jKeys:
        offsets[key] += values[key]

        window[key].Update(0)

    saveParams()

def moveAllJoints(ds):
    cnt = 0
    changed = True
    rad1 = 0.5 * math.pi / 180.0
    while changed and not stopMoving:
        changed = False
        for i, (d, a) in enumerate(zip(ds, Angles)):
            if rad1 < abs(d - a):
                changed = True
                Angles[i] += np.sign(d - a) * rad1
                setAngle(jKeys[i], Angles[i])

        showJoints(Angles)
        cnt += 1
        yield

    print("move end", cnt)

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

def calc(ts):
    j0, j1, j2, j3, j4, j5 = ts

    p1 = Vec2(0, L0)

    p2 = p1 + (Vec2(-L1,0).rot(j1))
    print(f'p2: %d %d' % (int(- p2.x), int(p2.y)))

    p3 = p2 + L2 * (p2 - p1).unit().rot(j2)
    print(f'p3: %d %d' % (int(- p3.x), int(p3.y)))

    p4 = p3 + L3 * (p3 - p2).unit().rot(j3 - 0.5 * np.pi)
    print(f'p4: %d %d' % (int(- p4.x), int(p4.y)))

    tcp = p4 + L4 * (p3 - p2).unit().rot(j3)
    print(f'tcp: %d %d' % (int(- tcp.x), int(tcp.y)))

    et = (p4 - tcp).unit()
    theta = np.arctan2(et.y, et.x)
    return arr([ -tcp.x, 0, tcp.y, 0, theta])

def IK(pose):
    x, y, z, phi, theta = pose

    r   = Vec2(x, y).len()

    p1 = Vec2(0, L0)

    tcp = Vec2(-r, z)
    p4 = tcp + L4 * Vec2(math.cos(theta), math.sin(theta))
    print(f'p4: %d %d' % (int(- p4.x), int(p4.y)))

    # tcpからp4に向かう単位ベクトル
    et = (p4 - tcp).unit()

    # p4からp3に向かう単位ベクトル
    e4 = et.rot(radian(-90))

    p3 = p4 + L3 * e4
    print(f'p3: %d %d' % (int(- p3.x), int(p3.y)))

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

    print('e31:%.2f %.2f e32:%.2f %.2f alpha:%.1f' % (e31.x, e31.y, e32.x, e32.y, degree(alpha)))

    p2 = p3 + L2 * e32
    print(f'p2: %d %d' % (int(- p2.x), int(p2.y)))

    ts = [0] * nax

    ts[0] = np.arctan2(y, x)

    ts[1] = (p2 - p1).arctan2p() - np.pi

    ts[2] = (p3 - p2).arctan2p() - (p2 - p1).arctan2p()

    ts[3] = (tcp - p4).arctan2p() - (p3 - p2).arctan2p()

    ts[4] = phi

    print("J2:%.1f  J3:%.1f  J4:%.1f " % (degree(ts[1]), degree(ts[2]), degree(ts[3])))

    return ts

def calc2(ts):
    tsum = 0.0
    
    for i, t in enumerate(ts):
        if i == 0:
            u = links[0]
            v = 0.0
                        
        elif i <= 3:
            
            l = links[i]
            tsum += t
            u += l * math.cos(tsum)
            v += l * math.sin(tsum)
    
    r = v
    x = r * math.cos(ts[0])
    y = r * math.sin(ts[0])
    z = u
    r2 = ts[4]
    
    return arr([x, y, z, tsum, r2])

def Jacob(ts, pos):
    m = np.zeros((nax-1, nax-1), dtype=np.float32)

    dt = 0.001
    for i in range(nax-1):
        ts2 = np.array(ts)
        ts2[i] += dt
        pos2 = calc(ts2)
        for j in range(nax-1):
            m[j, i] = (pos2[j] - pos[j]) / dt

    return m

def showPos(pos):
    for k, p in zip(["X", "Y", "Z"], pos[:3]):
        window[k].Update(p)
        
    for k, p in zip(["R1", "R2"], pos[3:]):
        window[k].Update(degree(p))
        
def move(dst):
    global Angles
    
    ts = arr(Angles)
    ts[-1] = dst[-1]
    src = calc(ts)
    
    dst = arr(dst)
    
    cnt = 100
    pos = src
    for idx in range(cnt):
        if stopMoving:
            break
        
        dst1 = src + (float(idx + 1) / float(cnt)) * (dst - src)
        
        dpos = dst1 - pos
        
        m = Jacob(ts, pos)
        mi = np.linalg.inv(m)
        
        dang = np.dot(mi, dpos[:-1])
        ts[:-1] += dang
        showJoints(ts)

        pos = calc(ts)
        showPos(pos)
        print("%d %s" % (idx, strPos(pos)))
        
        if idx % 10 == 0:
            yield

    moveAllJoints(ts)
        
def test():
    global angles
    
    a = np.array([[1,2],[3,4]])
    b = np.array([5, 6])
    c = np.dot(a,b)
    ai = np.linalg.inv(a)

    print("a = ", a)
    print("b = ", b)
    print("c = a * b =", c)
    print("ai = inv(a) =", ai)

    print("aixc =", np.dot(ai,c))

    showJoints([0.0] * nax)

    x, y, z, r1, r2 = calc(Angles)
    print("x:%.1f, y:%.1f, z:%.1f, r1:%.1f, r2:%.1f" % (x, y, z, r1, r2))

    t = 0.2 * math.pi
    x, y, z, r1, r2 = calc([t] * nax)
    print("x:%.1f, y:%.1f, z:%.1f, r1:%.1f, r2:%.1f" % (x, y, z, r1, r2))

    pos = calc(Angles)
    m = Jacob(Angles, pos)
    print("Jacob", pos, m)

    showJoints(Angles + 0.125 * math.pi)
    src = calc(Angles)
    m = Jacob(Angles, src)
    print("Jacob", m, np.linalg.inv(m))

loadParams()

if useSerial:
    ser = serial.Serial('COM5', 115200, timeout=0.1)


    # while True:
        
    #     print(ch)
    #     for c in ch:
    #         print(c, hex(ord(c)))

else:
    pwm = Adafruit_PCA9685.PCA9685(address=0x40)
    pwm.set_pwm_freq(60)

sg.theme('DarkAmber')   # SystemDefault Add a touch of color
# All the stuff inside your window.
layout = [
    [
    sg.Column([
        [ sg.Text('J1'), sg.Slider(range=(-120,130), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='J1') ],
        [ sg.Text('J2'), sg.Slider(range=(-120,130), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='J2') ],
        [ sg.Text('J3'), sg.Slider(range=(-120,130), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='J3') ],
        [ sg.Text('J4'), sg.Slider(range=(-120,130), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='J4') ],
        [ sg.Text('J5'), sg.Slider(range=(-120,130), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='J5') ],
        [ sg.Text('J6'), sg.Slider(range=(-120,130), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='J6') ],
    ]),
    sg.Column([
        [ sg.Text('X'), sg.Slider(range=(0,300), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='X') ],
        [ sg.Text('Y'), sg.Slider(range=(-300,300), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='Y') ],
        [ sg.Text('Z'), sg.Slider(range=(0,150), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='Z') ],
        [ sg.Text('R1'), sg.Slider(range=(-90,90), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='R1', resolution=1) ],
        [ sg.Text('R2'), sg.Slider(range=(0,120), default_value=0, size=(40,15), orientation='horizontal', change_submits=True, key='R2', resolution=1) ]
    ])
    ],
    [ sg.Button('Test'), sg.Button('Reset'), sg.Button('Ready'), sg.Button('Move'), sg.Button('Stop'), sg.Button('Home'), sg.Button('Cancel')]
]

# Create the Window
window = sg.Window('Robot Control', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    if moving is None:
        event, values = window.read()
    else:
        event, values = window.read(timeout=1)
        try:
            moving.__next__()
        except StopIteration:
            moving = None
            stopMoving = False
            print("stop moving")
        
    if event in jKeys:
        t = radian(values[event])
        setAngle(event, t)
        
        Angles = getAngles()
        pos = calc(Angles)
        showPos(pos)
        
    elif event == "Move":
        dst = getPos()
        ts = IK(dst)

        # if ts is not None:
        #     moving = moveAllJoints(ts)
        
    elif event == "Stop":
        stopMoving = True
        
    elif event == 'Home':
        setOffsets()
            
    elif event == "Test":
        test()
        src = calc(Angles)
        dst = calc(Angles + 0.125 * math.pi)

        print("src", src)
        print("dst", dst)

        move(dst)
        
    elif event == "Reset":
        for i, key in enumerate(jKeys):
            Angles[i] = 0
            setAngle(key, 0)

        showJoints([0] * nax)
        # ts = [0] * nax
        # moving = moveAllJoints(ts)

        if False:
            pos = calc(ts)
            showPos(pos)
            print("move end")
        
    elif event == "Ready":
        ts = [radian(5.0)] * nax
        moving = moveAllJoints(ts)
        
        if False:
            pos = calc(ts)
            showPos(pos)
            print("move end")
        
    elif event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    
window.close()

