### arucoマーカーを生成して、画像として保存する
import time
import cv2
from cv2 import aruco
import os
import numpy as np

import PySimpleGUI as sg
import cv2
from sklearn.linear_model import LinearRegression

from util import degree, read_params, Vec3


def make_markers():
    ### --- parameter --- ###

    # マーカーの保存先
    dir_mark = r'data/marker'

    # 生成するマーカー用のパラメータ
    cols = 5
    num_mark = cols * cols #個数
    size_mark = 500 #マーカーのサイズ

    ### --- マーカーを生成して保存する --- ###
    # マーカー種類を呼び出し
    dict_aruco = aruco.Dictionary_get(aruco.DICT_4X4_50)

    mg = size_mark // 2
    bg = np.full(( cols * (mg + size_mark) + mg, cols * (mg + size_mark) + mg), 255, dtype=np.uint8)
    for count in range(num_mark) :

        id_mark = count #countをidとして流用
        img_mark = aruco.drawMarker(dict_aruco, id_mark, size_mark)

        row = count  % cols
        col = count // cols

        y = mg + row * (size_mark + mg)
        x = mg + col * (size_mark + mg)

        bg[y:y+size_mark, x:x+size_mark] = img_mark

    for iy in range(2 * cols + 1):
        for ix in range(2 * cols + 1):

            if ix % 2 == 1 and iy % 2 == 1:
                continue

            x = mg // 2 + ix * (size_mark + mg) // 2
            y = mg // 2 + iy * (size_mark + mg) // 2

            mg2 = mg // 8
            cv2.line(bg, (x - mg2, y), (x + mg2, y), (0,))
            cv2.line(bg, (x, y - mg2), (x, y + mg2), (0,))

    # bg = make_dark(bg, gray_scale)
    cv2.imwrite(f'{dir_mark}/markers4x4.png', bg)        


def init_markers(params):
    global dictionary, camera_matrix, dist_coeffs

    dictionary_type = aruco.DICT_4X4_50
    dictionary_name = '4X4_50'
    dictionary = aruco.getPredefinedDictionary(dictionary_type)

    camera_inf = params['cameras'][dictionary_name]

    camera_matrix = np.array(camera_inf['camera-matrix'])
    dist_coeffs = np.array( camera_inf['dist-coeffs'] )

def detect_markers(marker_ids, frame):
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)

    aruco.drawDetectedMarkers(frame, corners, ids)            

    marker_length = 13.5
    
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

if __name__ == '__main__':
    make_markers()