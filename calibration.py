import time
import math
import numpy as np
import PySimpleGUI as sg
import json
import cv2
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand, getCameraFrame
from util import jKeys, getGlb, radian, writeParams, t_all, spin, degree, Vec2
from s_curve import SCurve
from infer import Inference

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
            
            assert False, 'dst_rad = IK2(x, y, True)'

            if dst_rad is None:
                print(f'skip move x:{x} y:{y}')
                continue
            
            dst_deg = [degree(rad) for rad in dst_rad]
            mv = waitMoveAllJoints(dst_deg)
            while mv.__next__():
                yield

            print("move end")
            start_time = time.time()
            while time.time() - start_time < 4:
                yield

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

    writeParams(params)

    fitRegression(eye_xy, hand_x, hand_y)
