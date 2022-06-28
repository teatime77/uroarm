import sys
import time
import math
import serial
import cv2
import PySimpleGUI as sg
from util import writeParams, loadParams, spin
from camera import initCamera, getCameraFrame, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand
from calibration import Calibrate, get_markers, set_hsv_range
import numpy as np

def setAngle(channel : int, degree : float):
    cmd = "%d,%.1f\r" % (channel, degree)

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

down_pos = None
radius = None

def printCoor(event,x,y,flags,param):
    global down_pos, radius

    if event == cv2.EVENT_LBUTTONDOWN:
        print('down', x, y)
        down_pos = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        print('up'  , x, y)
        down_pos = None
        radius   = None

    elif event == cv2.EVENT_MOUSEMOVE and down_pos is not None:
        dx = down_pos[0] - x
        dy = down_pos[1] - y
        radius = int(math.sqrt(dx * dx + dy * dy))


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('COMを指定してください。')
        print('python servo.py COM*')
        sys.exit(0)

    com_port = sys.argv[1]

    try:
        ser = serial.Serial(com_port, 115200, timeout=1, write_timeout=1)
    except serial.serialutil.SerialException: 
        print(f'指定されたシリアルポートがありません。{com_port}')
        sys.exit(0)

    params, com_port, offsets, scales, Angles = loadParams()

    initCamera()

    layout = [
        [
            sg.TabGroup([
                [
                    sg.Tab('marker1', [
                        spin('H lo', '-Hlo1-',  95, 0, 180, False),
                        spin('H hi', '-Hhi1-',  99, 0, 180, False),
                        spin('S lo', '-Slo1-', 125, 0, 255, False),
                        spin('V lo', '-Vlo1-',   90, 0, 255, False)
                    ])
                    , 
                    sg.Tab('marker2', [
                        spin('H lo', '-Hlo2-',  19, 0, 180, False),
                        spin('H hi', '-Hhi2-',  23, 0, 180, False),
                        spin('S lo', '-Slo2-', 190, 0, 255, False),
                        spin('V lo', '-Vlo2-',   90, 0, 255, False)
                    ])
                ]
            ], key='-markers-')
            ,
            sg.Column([
                spin('J1', 'J1', 90, 0, 270, True),
                spin('J2', 'J2', 90, 0, 270, True),
                spin('J3', 'J3', 90, 0, 270, True),
                spin('J4', 'J4', 90, 0, 270, True),
                spin('J5', 'J5', 90, 0, 270, True),
                spin('J6', 'J6', 90, 0, 270, True)
            ])
        ]
        ,
        [ sg.Button('Close')]
    ]

    window = sg.Window('Servomotor', layout, disable_minimize=True, element_justification='c')

    last_capture = time.time()
    is_first = True
    while True:
        event, values = window.read(timeout=1)
            
        jKeys = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

        if event in jKeys:
            channel = jKeys.index(event)
            degree  = float(values[event])

            setAngle(channel, degree)

        elif event == sg.WIN_CLOSED or event == 'Close':
            writeParams(params)

            closeCamera()
            break

        elif event == "Calibrate":
            moving = Calibrate(params, values)        

        else:
            if 0.1 < time.time() - last_capture:
                last_capture = time.time()

                frame = getCameraFrame()

                if radius is None:
                    frame, (rx, ry), (bx, by) = get_markers(values, frame)

                else:
                    if window['-markers-'].get() == 'marker1':
                        marker_idx = 1 
                    else:
                        marker_idx = 2

                    frame = set_hsv_range(window, marker_idx, frame, down_pos, radius)
                cv2.imshow("camera", frame)

                if is_first:

                    is_first = False
                
                    cv2.setMouseCallback('camera', printCoor)

    window.close()

