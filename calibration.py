import time
import math
import numpy as np
import PySimpleGUI as sg
import json
import cv2
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand, getCameraFrame
from util import nax, jKeys, servo_angle_keys, getGlb, radian, write_params, t_all, spin, spin2, degree, Vec2, sleep
from infer import Inference
from kinematics import move_linear_ok, move_xyz

def showMark(values, frame, idx: int):
    h_lo = float(values[f'-Hlo{idx + 1}-'])
    h_hi = float(values[f'-Hhi{idx + 1}-'])
    s_lo = float(values[f'-Slo{idx + 1}-'])
    v_lo = float(values[f'-Vlo{idx + 1}-'])

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    h = img_hsv[:,:,0]
    s = img_hsv[:,:,1]
    v = img_hsv[:,:,2]

    h = (h.astype(np.int32) - 80) % 180

    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[ (h_lo <= h) & (h <= h_hi) & (s_lo <= s) & (v_lo <= v) ] = 255    

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        areas = [ cv2.contourArea(cont) for cont in contours ]

        max_index = areas.index( max(areas) )
        cont = contours[max_index]

        cv2.drawContours(frame, contours, max_index, (255,255,255), -1)

        M = cv2.moments(cont)

        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_8)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 0, 0), thickness=2, lineType=cv2.LINE_8)

            return Vec2(cx, -cy)

    return None

def get_markers(window, values, frame):
    
    centers = [ showMark(values, frame, i) for i in range(3) ]

    if all(x is not None for x in centers):
        v1 = (centers[1] - centers[0]).unit()
        v2 = (centers[2] - centers[1]).unit()

        ip = max(-1, min(1, v1.dot(v2)))
        rad = math.acos(ip)
        assert 0 <= rad and rad <= math.pi

        marker_deg = degree(rad)
        if v1.cross(v2) < 0:
            marker_deg = -marker_deg

        window['-rotation-'].update(value='%.1f' % marker_deg)
    else:
        marker_deg = None
        window['-rotation-'].update(value='')

    return frame, marker_deg


def set_hsv_range(params, window, marker_idx, frame, center, radius):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)

    img_hsv[:,:,0] = (img_hsv[:,:,0] - 80) % 180

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, (255,), thickness=-1) 

    hsv = [ np.where(mask == 0, np.nan, img_hsv[:, :, i]) for i in range(3)]

    h_avg, s_avg, v_avg = [ np.nanmean(x) for x in hsv ]
    h_sdt, s_sdt, v_sdt = [ np.nanstd(x) for x in hsv ]

    cv2.circle(frame, center, radius, (255,255,255), thickness=2) 

    h_lo = round(h_avg - 3 * h_sdt)
    h_hi = round(h_avg + 3 * h_sdt)
    s_lo = round(s_avg - 3 * s_sdt)
    v_lo = round(v_avg - 3 * v_sdt)

    window[f'-Hlo{marker_idx + 1}-'].update(value=h_lo)
    window[f'-Hhi{marker_idx + 1}-'].update(value=h_hi)
    window[f'-Slo{marker_idx + 1}-'].update(value=s_lo)
    window[f'-Vlo{marker_idx + 1}-'].update(value=v_lo)

    params["markers"][marker_idx] = {
        "h-lo": h_lo,
        "h-hi": h_hi,
        "s-lo": s_lo,
        "v-lo": v_lo
    }

    return frame                   

def fitRegression(eye_xy, hand_x, hand_y):
    X = np.array(eye_xy)
    hand_x = np.array(hand_x)
    hand_y = np.array(hand_y)

    glb = getGlb()
    glb.regX = LinearRegression().fit(X, hand_x)

    glb.regY = LinearRegression().fit(X, hand_y)

    print('reg x', glb.regX.coef_)
    print('reg y', glb.regY.coef_)

    prd_x = glb.regX.predict(X)
    prd_y = glb.regY.predict(X)
    print(f'X:{X.shape} arm-x:{hand_x.shape} arm-y:{hand_y.shape} prd-x:{prd_x.shape} prd-y:{prd_y.shape}')

    for i in range(X.shape[0]):
        print(f'cam:{X[i, 0]} {X[i, 1]} arm:{hand_x[i]} {hand_y[i]} prd:{int(prd_x[i]) - hand_x[i]} {int(prd_y[i]) - hand_y[i]}')



def Calibrate(params, values):
    eye_xy = []
    hand_x = []
    hand_y = []

    for x in [150, 200, 250, 300]:
        for y in [-50, 0, 50]:
            print(f'start move x:{x} y:{y}')
            
            for _ in move_xyz(x, y, 30):
                yield

            if not move_linear_ok:
                print(f'skip move x:{x} y:{y}')
                continue
            
            print("move end")
            for _ in sleep(4):
                yield

            centers = [ showMark(values, frame, i) for i in range(2) ]

            (rx, ry), (bx, by) = get_markers(values)

            if bx is not None and rx is not None:
                CamX = 0.5 * (bx + rx)
                CamY = 0.5 * (by + ry)


            print(f'hand eye x:{x} y:{y} cx:{camX()} cy:{camY()}')

            eye_xy.append([ camX(), camY() ])
            hand_x.append(x)
            hand_y.append(y)

    params['calibration'] = {
        'eye-xy' : eye_xy,
        'hand-x' : hand_x,
        'hand-y' : hand_y
    }

    write_params(params)

    fitRegression(eye_xy, hand_x, hand_y)


def draw_grid(frame):
    h, w, _ = frame.shape

    for y in range(0, h, h // 20):
        cv2.line(frame, (0, y), (w, y), (255, 0, 0))

    for x in range(0, w, w // 20):
        cv2.line(frame, (x, 0), (x, h), (255, 0, 0))

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

if __name__ == '__main__':

    params, servo_angles, Angles = loadParams()
    marker1, marker2, marker3 = params['markers']

    init_servo(params, servo_angles, Angles)

    initCamera()

    layout = [
        [
            sg.Column([
                [
                    sg.TabGroup([
                        [
                            sg.Tab('marker1', [
                                spin('H lo', '-Hlo1-', marker1['h-lo'], 0, 180, False),
                                spin('H hi', '-Hhi1-', marker1['h-hi'], 0, 180, False),
                                spin('S lo', '-Slo1-', marker1['s-lo'], 0, 255, False),
                                spin('V lo', '-Vlo1-', marker1['v-lo'], 0, 255, False)
                            ])
                            , 
                            sg.Tab('marker2', [
                                spin('H lo', '-Hlo2-', marker2['h-lo'], 0, 180, False),
                                spin('H hi', '-Hhi2-', marker2['h-hi'], 0, 180, False),
                                spin('S lo', '-Slo2-', marker2['s-lo'], 0, 255, False),
                                spin('V lo', '-Vlo2-', marker2['v-lo'], 0, 255, False)
                            ])
                            , 
                            sg.Tab('marker3', [
                                spin('H lo', '-Hlo3-', marker3['h-lo'], 0, 180, False),
                                spin('H hi', '-Hhi3-', marker3['h-hi'], 0, 180, False),
                                spin('S lo', '-Slo3-', marker3['s-lo'], 0, 255, False),
                                spin('V lo', '-Vlo3-', marker3['v-lo'], 0, 255, False)
                            ])
                        ]
                    ], key='-markers-')
                ]
                ,
                [
                    sg.Text('', key='-rotation-')
                ]
            ])
            ,
            sg.Column([
                spin2(f'J{i+1}', f'J{i+1}', servo_angles[i], degree(Angles[i]), -120, 120, True) + [sg.Button('start', key=f'-start-J{i+1}-')]
                for i in range(nax)
            ])
        ]
        ,
        [ sg.Checkbox('grid', default=False, key='-show-grid-'), sg.Button('Reset'), sg.Button('Close')]
    ]

    window = sg.Window('calibration', layout, disable_minimize=True, element_justification='c')

    start_keys = [ f'-start-J{i+1}-' for i in range(nax) ]

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

            moving = move_servo(ch, deg)

            window[jKeys[ch]].update(value=int(servo_to_angle(ch, deg)))

        elif event in jKeys:
            ch = jKeys.index(event)
            deg = float(values[event])

            moving = move_joint(ch, deg)

            window[servo_angle_keys[ch]].update(value=int(angle_to_servo(ch, deg)))

        elif event in start_keys:
            moving = calibrate_angle(event)

        elif event == 'Reset':
            for ch in range(nax):
                set_servo_angle(ch, 90)
                window[f'J{ch + 1}-servo'].update(value=90)

        elif event == sg.WIN_CLOSED or event == 'Close':

            params['servo-angles'] = servo_angles
            write_params(params)

            closeCamera()
            break

        elif event == "Calibrate":
            moving = Calibrate(params, values)        

        else:
            if 0.1 < time.time() - last_capture:
                last_capture = time.time()

                frame = getCameraFrame()

                if radius is None:
                    frame, marker_deg = get_markers(window, values, frame)

                else:
                    marker_idx = [ 'marker1', 'marker2', 'marker3' ].index( window['-markers-'].get() )

                    frame = set_hsv_range(params, window, marker_idx, frame, down_pos, radius)

                if values['-show-grid-']:
                    draw_grid(frame)

                cv2.imshow("camera", frame)

                if is_first:

                    is_first = False
                
                    cv2.setMouseCallback('camera', mouse_callback)
