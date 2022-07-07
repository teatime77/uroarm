"""arucoマーカーを生成して、画像として保存する

以下を参考にした。
    naoki-mizuno/charuco_webcam.py
    https://gist.github.com/naoki-mizuno/c80e909be82434ddae202ff52ea1f80a
"""
### 
import time
import cv2
from cv2 import aruco
import os
import numpy as np

import PySimpleGUI as sg
import cv2
import subprocess

from util import write_params, read_params
from camera import initCamera, closeCamera, getCameraFrame

def draw_axis(frame, camera_matrix, dist_coeffs, board, verbose=True):
    corners, ids, rejected_points = cv2.aruco.detectMarkers(frame, dictionary)
    if corners is None or ids is None:
        return None
    if len(corners) != len(ids) or len(corners) == 0:
        return None

    try:
        ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners,
                                                                    ids,
                                                                    frame,
                                                                    board)
        ret, p_rvec, p_tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners,
                                                                c_ids,
                                                                board,
                                                                camera_matrix,
                                                                dist_coeffs)
        if p_rvec is None or p_tvec is None:
            return None
        if np.isnan(p_rvec).any() or np.isnan(p_tvec).any():
            return None
        cv2.aruco.drawAxis(frame,
                        camera_matrix,
                        dist_coeffs,
                        p_rvec,
                        p_tvec,
                        0.1)
        # cv2.aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # cv2.aruco.drawDetectedMarkers(frame, rejected_points, borderColor=(100, 0, 240))
    except cv2.error:
        return None

    if verbose:
        print('Translation : {0}'.format(p_tvec))
        print('Rotation    : {0}'.format(p_rvec))
        print('Distance from camera: {0} m'.format(np.linalg.norm(p_tvec)))

    return frame

if __name__ == '__main__':


    initCamera()

    dictionary_name = '4X4_50'
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50) #  DICT_5X5_50

    squares_x = 5
    squares_y = 7
    square_length = 0.0325
    marker_length = 0.01625
    board = aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)

    layout = [
        [ sg.Button('Add'), sg.Button('Finish'), sg.Button('Close') ]
    ]

    window = sg.Window('marker', layout, disable_minimize=True, element_justification='c')

    all_corners = []
    all_ids = []

    camera_matrix   = None
    dist_coeffs      = None

    last_capture = time.time()
    while True:
        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED or event == 'Close':

            closeCamera()
            break
        
        else:

            frame = getCameraFrame()

            if camera_matrix is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                imsize = gray.shape

                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, dictionary)

                if len(corners) > 0:
                    ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                    # ret is the number of detected corners
                    if ret > 0:

                        aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids, (255, 0, 0))
                        if len(c_corners) == (squares_x - 1) * (squares_y - 1) and 0.5 <= time.time() - last_capture:
                            all_corners.append(c_corners)
                            all_ids.append(c_ids)

                            print(len(c_corners), len(c_ids), len(all_corners))

                            last_capture = time.time()

                            if len(all_corners) == 50:
                                print('calibrate camera')
                                ret, camera_matrix, dist_coeffs, rvec, tvec = cv2.aruco.calibrateCameraCharuco(
                                    all_corners, all_ids, board, imsize, None, None
                                )

                                params = read_params()
                                params['cameras'][dictionary_name] = {
                                    "camera-matrix": camera_matrix.tolist(),
                                    "dist-coeffs": dist_coeffs.tolist()
                                }

                                write_params(params)
                                print('write params')
                                break



                else:
                    c_corners   = None
                    c_ids       = None

            else:
                draw_axis(frame, camera_matrix, dist_coeffs, board, False)


            cv2.imshow("camera", frame)

    closeCamera()    
