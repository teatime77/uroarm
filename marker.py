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

    # aruco.drawDetectedMarkers(frame, corners, ids)            

    marker_length = 15
    
    vecs = [None] * len(marker_ids)

    if corners is None or ids is None:
        return frame, vecs

    for corner, id in zip(corners, ids):

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, dist_coeffs)

        # Remove single-dimensional entries from the shape of an array.
        tvec = np.squeeze(tvec)
        rvec = np.squeeze(rvec)

        # frame = aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 10)
        
        assert corner.shape == (1, 4, 2)
        corner = np.squeeze(corner)
        assert corner.shape == (4, 2)

        left_top = corner.min(axis=0)

        pxy = corner.mean(axis=0)

        cv2.putText(frame, text=f'{id[0]}', org=(left_top[0], left_top[1]),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            color=(0, 0, 255),
            thickness=4,
            lineType=cv2.LINE_AA
        )

        tvec[1] = - tvec[1]

        if id in marker_ids:

            idx = marker_ids.index(id)

            vecs[idx] = np.concatenate([tvec, pxy])

    return frame, vecs
