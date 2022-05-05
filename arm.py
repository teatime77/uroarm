import time
import math
import numpy as np
import PySimpleGUI as sg
import json
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY
from util import getGlb

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
poseDim = 5
moving = None
stopMoving = False
Angles = [0] * nax
dstAngles = [0] * nax
moveCnt = 30

zs = arr([ 0.5, 11.0, 17.5, 23.5, 28.8 ]) - 0.5
links = [ zs[i+1] - zs[i] for i in range(4) ]

L0, L1, L2, L3, L4 = [ 111, 105, 98, 25, 162 ]

jKeys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
posKeys = [ "X", "Y", "Z", "R1", "R2" ]


def degree(t):
    if type(t) is list:
        return [ x * 180.0 / math.pi for x in t ]
    else:
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

def getPose():
    dst = [ float(values[k]) for k in posKeys ]
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


def setAngle(event, deg):
    global Angles

    ch = jKeys.index(event)

    Angles[ch] = radian(deg)

    deg += offsets[event]

    if event == "J2":
        deg *= -1

    v = [ 90, 97, 105, 100, 90, 90 ]

    deg *= float(v[ch]) / 90.0

    deg += 90

    cmd = "%d,%.1f\r" % (ch, deg)

    while True:
        try:
            n = ser.write(cmd.encode('utf-8'))
            break
        except serial.SerialTimeoutException:
            print("write time out")
            time.sleep(1)

    ret = ser.read_all().decode('utf-8')
    if "error" in ret:
        print("read", ret)

def getAngles():
    return [radian(values[x]) for x in jKeys]

def jointKey(i):
    return "J%d" % (i+1)

def showJoints(ts):
    for i, j in enumerate(ts):
        key = jointKey(i)
        window[key].Update(int(round(degree(j))))

def setOffsets():
    for key in jKeys:
        offsets[key] += values[key]

        window[key].Update(0)

    saveParams()

def moveAllJoints(dsts):
    for dst in dsts:
        src = [degree(x) for x in Angles]

        start_time = time.time()
        for i in range(moveCnt):

            r = float(i + 1) / float(moveCnt)

            for j in range(nax):
                deg = (1.0 - r) * src[j] + r * dst[j]
                setAngle(jKeys[j], deg)

            # showJoints(Angles)
            yield

        print("move end %d msec" % int(1000 * (time.time() - start_time) / moveCnt))


def Calibrate():
    cam_xy = []
    arm_x = []
    arm_y = []

    for x in [150, 200, 250, 300]:
        for y in [-50, 0, 50]:
            print(f'start move x:{x} y:{y}')
            dst_rad = IK2(x, y, True)

            if dst_rad is None:
                print(f'skip move x:{x} y:{y}')
                continue
            
            dst_deg = [degree(rad) for rad in dst_rad]
            src_deg = [degree(rad) for rad in Angles]

            cnt = 30
            for i in range(cnt):
                
                r = float(i + 1) / float(cnt)

                for j in range(nax):
                    deg = (1.0 - r) * src_deg[j] + r * dst_deg[j]
                    setAngle(jKeys[j], deg)

                # showJoints(Angles)

                yield

            print("move end")
            start_time = time.time()
            while time.time() - start_time < 4:
                yield
            print(f'hand eye x:{x} y:{y} cx:{camX()} cy:{camY()}')

            cam_xy.append([ camX(), camY() ])
            arm_x.append(x)
            arm_y.append(y)

    X = np.array(cam_xy)
    arm_x = np.array(arm_x)
    arm_y = np.array(arm_y)

    glb = getGlb()
    glb.regX = LinearRegression().fit(X, arm_x)

    glb.regY = LinearRegression().fit(X, arm_y)

    print('reg x', glb.regX.coef_)
    print('reg y', glb.regY.coef_)

    prd_x = glb.regX.predict(X)
    prd_y = glb.regY.predict(X)
    print(f'X:{X.shape} arm-x:{arm_x.shape} arm-y:{arm_y.shape} prd-x:{prd_x.shape} prd-y:{prd_y.shape}')

    for i in range(X.shape[0]):
        print(f'cam:{X[i, 0]} {X[i, 1]} arm:{arm_x[i]} {arm_y[i]} prd:{int(prd_x[i]) - arm_x[i]} {int(prd_y[i]) - arm_y[i]}')

def moveIK():
    # 初期ポーズ
    src = calc(Angles)

    # 目標ポーズ
    dst = getPose()

    cnt = 100
    for step in range(cnt):

        r = float(step + 1) / float(cnt)

        # 途中のポーズ
        pose = [0] * poseDim
        for j in range(poseDim):
            pose[j] = (1.0 - r) * src[j] + r * dst[j]

        # 逆運動学
        rads = IK(pose)

        for j, rad in enumerate(rads):
            # Angles[j] = rad
            setAngle(jKeys[j], degree(rad))

            showJoints(Angles)

        yield

    print([degree(x) for x in Angles])
    print("move end")




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

def calc(rads):
    j0, j1, j2, j3, j4, j5 = rads

    p1 = Vec2(0, L0)

    p2 = p1 + (Vec2(-L1,0).rot(j1))
    # print(f'p2: %d %d' % (int(- p2.x), int(p2.y)))

    p3 = p2 + L2 * (p2 - p1).unit().rot(j2)
    # print(f'p3: %d %d' % (int(- p3.x), int(p3.y)))

    p4 = p3 + L3 * (p3 - p2).unit().rot(j3 - 0.5 * np.pi)
    # print(f'p4: %d %d' % (int(- p4.x), int(p4.y)))

    tcp = p4 + L4 * (p3 - p2).unit().rot(j3)
    # print(f'tcp: %d %d' % (int(- tcp.x), int(tcp.y)))

    et = (p4 - tcp).unit()
    theta = np.arctan2(et.y, et.x)

    r = tcp.len()

    x = r * math.cos(j0)
    y = r * math.sin(j0)
    z = tcp.y

    # print('r:%.1f j0:%.1f x:%.1f y:%.1f z:%.1f' % (r, degree(j0), x, y, z))

    return arr([ x, y, z, j4, theta])

def IK2(x, y, is_down):
    r   = Vec2(x, y).len()

    J123_x = J123_x_down if is_down else J123_x_up

    xs = [ x_ for _, _, x_ in J123_x ]
    min_x = min(xs)
    max_x = max(xs)
    if r < min_x or max_x < r:
        return None

    diff_x = [ abs(x_ - r) for x_ in xs ]
    i = diff_x.index(min(diff_x))

    deg1, deg23, _ = J123_x[i]

    ts = [0] * nax

    ts[0] = np.arctan2(y, x)

    ts[1] = radian(deg1)

    ts[2] = radian(deg23)

    ts[3] = ts[2]

    ts[4] = ts[0]

    ts[5] = Angles[5]

    return ts

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
        window[k].Update(int(round(p)))
        
    for k, p in zip(["R1", "R2"], pos[3:]):
        window[k].Update(int(round(degree(p))))
        
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
        
        if idx % 10 == 0:
            yield

    moveAllJoints([ts])
        
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

def naturalPose():
    global J123_x_down, J123_x_up
    j0, j4, j5 = [ 0, 0, 0]

    print("start")

    J123_x_down = []
    J123_x_up   = []

    for is_down in [ True, False]:

        if is_down:
            J123_x = J123_x_down
            file_name = 'J123-x-down.csv'
            target_z = 10

        else:
            J123_x = J123_x_up
            file_name = 'J123-x-up.csv'
            target_z = 40
            
        with open(file_name, 'w') as f:
            f.write('J1,J23,x,diff\n')

            min_deg1 = None

            for deg23 in np.linspace(0, 90, 90 * 5):
                j23 = radian(deg23)
                
                min_diff = 1000
                min_x = 0

                if min_deg1 is None:
                    deg1_list = np.linspace(-90, 0, 90 * 5)
                else:
                    deg1_list = np.linspace(min_deg1 - 1, min_deg1 + 1, 10)

                for deg1 in deg1_list:
                    j1 = radian(deg1)
                    rads = [ j0, j1, j23, j23, j4, j5 ]

                    x, y, z, _, _ = calc(rads).tolist()

                    diff = abs(z - target_z)
                    if diff < min_diff:
                        min_diff = diff
                        min_deg1 = deg1
                        min_x    = x

                    # elif min_diff < 1:
                    #     break

                if min_diff < 1:
                    J123_x.append([min_deg1, deg23, min_x])
                    f.write('%.1f,%.1f,%.1f,%.1f\n' % (min_deg1, deg23, min_x, min_diff))
    print("end")

H_lo =  80
H_hi = 140
S_lo = 170
S_hi = 255
V_lo = 180
V_hi = 255

def spin(label, key, val, min_val, max_val, bind_return_key = True):

    return [ 
        sg.Text(label),
        sg.Spin(list(range(min_val, max_val + 1)), initial_value=val, size=(5, 1), key=key, enable_events=not bind_return_key, bind_return_key=bind_return_key )
    ]

def openHand():
    setAngle('J6', -25)

def closeHand():
    setAngle('J6',  20)

def grabWork(x, y):
    x += 20
    openHand()

    dst_rad = IK2(x, y, True)

    if dst_rad is None:
        print(f'skip move x:{x} y:{y}')
        return
    
    dst_deg = [degree(rad) for rad in dst_rad]
    src_deg = [degree(rad) for rad in Angles]

    cnt = 30
    for i in range(cnt):
        
        r = float(i + 1) / float(cnt)

        for j in range(nax):
            deg = (1.0 - r) * src_deg[j] + r * dst_deg[j]
            setAngle(jKeys[j], deg)

        # showJoints(Angles)

        yield

    closeHand()
    
    start_time = time.time()
    while time.time() - start_time < 4:
        yield

    setAngle('J2', -60)

naturalPose()

loadParams()

initCamera()

if useSerial:
    ser = serial.Serial('COM3', 115200, timeout=1, write_timeout=1)


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
        spin('H lo', '-Hlo-', H_lo, 0, 180, False) + spin('H hi', '-Hhi-', H_hi, 0, 180, False),
        spin('S lo', '-Slo-', S_lo, 0, 255, False) + spin('S hi', '-Shi-', S_hi, 0, 255, False),
        spin('V lo', '-Vlo-', V_lo, 0, 255, False) + spin('V hi', '-Vhi-', V_hi, 0, 255, False)
    ]),
    sg.Column([
        spin('J1', 'J1', 0, -120, 130),
        spin('J2', 'J2', 0, -120, 130),
        spin('J3', 'J3', 0, -120, 130),
        spin('J4', 'J4', 0, -120, 130),
        spin('J5', 'J5', 0, -120, 130),
        spin('J6', 'J6', 0, -120, 130)
    ]),
    sg.Column([
        spin('X', 'X' , 0,    0, 400 ),
        spin('Y', 'Y' , 0, -300, 300 ),
        spin('Z', 'Z' , 0,    0, 150 ),
        spin('R1', 'R1', 0, -90,  90 ),
        spin('R2', 'R2', 0,   0, 120 )
    ])
    ],
    [ sg.Button('Test'), sg.Button('Reset'), sg.Button('Ready'), sg.Button('Move'), sg.Button('Stop'), sg.Button('Home'), sg.Button('Send'), sg.Button('Calibrate'), sg.Button('Cancel')]
]

# Create the Window
window = sg.Window('Robot Control', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    if moving is None:
        event, values = window.read(timeout=1)
    else:
        event, values = window.read(timeout=1)
        try:
            moving.__next__()
        except StopIteration:
            moving = None
            stopMoving = False

            showJoints(Angles)

            pos = calc(Angles)
            showPos(pos)

            print("stop moving")
        
    if event in jKeys:
        degs = [ float(values[key]) for key in jKeys ]
        moving = moveAllJoints([ degs ])
        
    elif event == "Move" or event in posKeys:
        # 目標ポーズ
        pose = getPose()

        x, y, z, phi, theta = pose
        rads_down = IK2(x, y, True)
        rads_up   = IK2(x, y, False)

        if rads_down is None or rads_up is None:
            continue
        
        degs_down = [ degree(rad) for rad in rads_down ]
        degs_up   = [ degree(rad) for rad in rads_up ]

        moving = moveAllJoints([degs_up, degs_down])

        # 逆運動学
        # rads = IK(pose)
        # moving = moveAllJoints([degs])

    elif event == "Calibrate":
        moving = Calibrate()        

    elif event == "Stop":
        moving = None
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
        degs = [0] * nax
        degs[5] = degree(Angles[5])
        moving = moveAllJoints([degs])
        
    elif event == "Ready":
        degs = [ 0, -60, 66, 70, 0, 0  ]
        moving = moveAllJoints([degs])

    elif event == "Send":
        sendImage(values)

    elif event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        closeCamera()
        break

    else:
        if moving is None:
            readCamera(values)

            glb = getGlb()
            if glb.prdX is not None:
                moving = grabWork(glb.prdX, glb.prdY)

                glb.prdX, glb.prdY = (None, None)
    
window.close()

