### arucoマーカーを生成して、画像として保存する
import sys
import time
import cv2
from cv2 import aruco
import os
import numpy as np

import PySimpleGUI as sg
import cv2
from sklearn.linear_model import LinearRegression

def init_markers(params):
    global dictionary, camera_matrix, dist_coeffs

    dictionary_type = aruco.DICT_4X4_50
    dictionary_name = '4X4_50'
    dictionary = aruco.getPredefinedDictionary(dictionary_type)

    if not 'cameras' in params or not dictionary_name in params['cameras']:
        print('No camera calibration')
        sys.exit(0)

    camera_inf = params['cameras'][dictionary_name]

    camera_matrix = np.array(camera_inf['camera-matrix'])
    dist_coeffs = np.array( camera_inf['dist-coeffs'] )

def detect_markers(marker_ids, frame):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)

    aruco.drawDetectedMarkers(frame, corners, ids)            

    marker_length = 15
    
    vecs = [None] * len(marker_ids)

    if corners is None or ids is None:
        return frame, vecs

    for corner, id in zip(corners, ids):
        if not id in marker_ids:
            continue

        idx = marker_ids.index(id)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, dist_coeffs)

        # 不要なaxisを除去
        tvec = np.squeeze(tvec)
        rvec = np.squeeze(rvec)

        frame = aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 10)
        
        assert corner.shape == (1, 4, 2)
        corner = np.squeeze(corner)
        assert corner.shape == (4, 2)

        pxy = corner.mean(axis=0)

        tvec[1] = - tvec[1]

        vecs[idx] = np.concatenate([tvec, pxy])

    return frame, vecs
