import time
import math
import numpy as np
import PySimpleGUI as sg
import json
import cv2
from sklearn.linear_model import LinearRegression
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand, getCameraFrame
from util import jKeys, getGlb, radian, writeParams, loadParams, t_all, spin, degree
from s_curve import SCurve
from infer import Inference

def mask_by_cont(shape, cont):
    mask = np.zeros(shape, np.uint8)
    cv2.drawContours(mask, [cont], 0, 255, -1)

    assert mask.dtype == np.uint8
    return mask

def showMark(values, frame, color):
    if color == 'red':
        h_lo = float(values['-Hlo1-'])
        h_hi = float(values['-Hhi1-'])
        s_lo = float(values['-Slo1-'])
        v_lo = float(values['-Vlo1-'])
    elif color == 'blue':
        h_lo = float(values['-Hlo2-'])
        h_hi = float(values['-Hhi2-'])
        s_lo = float(values['-Slo2-'])
        v_lo = float(values['-Vlo2-'])
    else:
        assert False

        v_lo = float(values['-Vlo-'])

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    h = img_hsv[:,:,0]
    s = img_hsv[:,:,1]
    v = img_hsv[:,:,2]

    h = (h.astype(np.int32) - 80) % 180

    mask = np.zeros(h.shape, dtype=np.uint8)
    mask[ (h_lo <= h) & (h <= h_hi) & (s_lo <= s) & (v_lo <= v) ] = 255    


    # f = cv2.inRange(img_hsv[:, :, 0], h_lo, h_hi)    

    # f = frame.astype(float)
    # f1 = 255.0 * f[:, :, color_idx] / (f[:, :, 0] + f[:, :, 1] + f[:, :, 2])
    # f = np.fmin(f1, 0.3 * f[:, :, color_idx])
    # f = f.astype(np.uint8)

    # if color_idx == 0:
    #     low = 100
    # else:
    #     low = 120

    # f = cv2.inRange(f, low, 255)    

    # frame[:, :, color_idx] = f
    # frame[:, :, 1] = 0
    # # frame[:, :, 2] = f

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # for cont in contours:
    #     cv2.drawContours(frame, [cont], 0, (255,255,255), -1)

    if len(contours) != 0:
        areas = [ cv2.contourArea(cont) for cont in contours ]

        # areas = [ cv2.mean(v, mask=mask_by_cont(v.shape, cont)) for cont in contours ]

        max_index = areas.index( max(areas) )
        cont = contours[max_index]

        cv2.drawContours(frame, contours, max_index, (255,255,255), -1)

        M = cv2.moments(cont)

        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # print(cv2.arcLength(cont,True))

            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_8)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 0, 0), thickness=2, lineType=cv2.LINE_8)

            # print(f'{color} area:{max(areas)} {cv2.contourArea(cont)} len:{cv2.arcLength(cont,True)}')
            return (cx, cy)

    return (None, None)

def get_markers(values, frame):
    
    # 赤玉の中心
    rx, ry = showMark(values, frame, 'red')

    # 青玉の中心
    bx, by = showMark(values, frame, 'blue')

    return frame, (rx, ry), (bx, by)



def set_hsv_range(window, marker_idx, frame, center, radius):
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)

    img_hsv[:,:,0] = (img_hsv[:,:,0] - 80) % 180

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, (255,), thickness=-1) 

    hsv = [ np.where(mask == 0, np.nan, img_hsv[:, :, i]) for i in range(3)]

    h_avg, s_avg, v_avg = [ np.nanmean(x) for x in hsv ]
    h_sdt, s_sdt, v_sdt = [ np.nanstd(x) for x in hsv ]

    cv2.circle(frame, center, radius, (255,255,255), thickness=2) 

    window[f'-Hlo{marker_idx}-'].update(value=round(h_avg - 3 * h_sdt))
    window[f'-Hhi{marker_idx}-'].update(value=round(h_avg + 3 * h_sdt))
    window[f'-Slo{marker_idx}-'].update(value=round(s_avg - 3 * s_sdt))
    window[f'-Vlo{marker_idx}-'].update(value=round(v_avg - 3 * v_sdt))

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


def calibrate_xy(params):
    cal = params['calibration']

    eye_xy = cal['eye-xy']
    hand_x = cal['hand-x']
    hand_y = cal['hand-y']

    fitRegression(eye_xy, hand_x, hand_y)
