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
from camera import initCamera, readCamera, closeCamera, sendImage, camX, camY, Eye2Hand, getCameraFrame

def make_dark(image, gray_scale):
    h, w = image.shape
    bgr = np.zeros((h, w, 3))
    for i in range(3):
        bgr[:,:,i] = np.where(image == 0, (i + 1) * gray_scale, 255)

    return bgr

def nxn(dt):
    if dt == aruco.DICT_4X4_50:
        return '4x4'
    else:
        return '5x5'

def make_board(dt):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(dt)

    squares_x = 5
    squares_y = 7
    square_length = 0.04
    marker_length = 0.02
    charucoBoard = aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)

    image_x = round(5 * 40 / 25.4 * 350)
    image_y = round(7 * 40 / 25.4 * 350)
    image = charucoBoard.draw((image_x, image_y))

    # bgr = make_dark(image, gray_scale)

    cv2.imwrite(f'data/marker/board{nxn(dt)}.png', image)

def make_markers(dt):
    ### --- parameter --- ###

    # マーカーの保存先
    dir_mark = r'data/marker'

    # 生成するマーカー用のパラメータ
    cols = 5
    num_mark = cols * cols #個数
    size_mark = 500 #マーカーのサイズ

    ### --- マーカーを生成して保存する --- ###
    # マーカー種類を呼び出し
    dict_aruco = aruco.Dictionary_get(dt)

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

    # bg = make_dark(bg, gray_scale)
    cv2.imwrite(f'{dir_mark}/markers{nxn(dt)}.jpg', bg)        

def make_board_and_markers():
    for dt in [aruco.DICT_4X4_50, aruco.DICT_5X5_50]:
        make_board(dt)
        make_markers(dt)

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


    msg = ''

    marker_length = 13.5
    
    vecs = [None] * len(marker_ids)

    if corners is None or ids is None:
        return frame, vecs

    height, width = frame.shape[:2]

    for corner, id in zip(corners, ids):
        if not id in marker_ids:
            continue

        idx = marker_ids.index(id)

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, dist_coeffs)

        # 不要なaxisを除去
        tvec = np.squeeze(tvec)
        rvec = np.squeeze(rvec)
        # 回転ベクトルからrodoriguesへ変換
        rvec_matrix = cv2.Rodrigues(rvec)
        rvec_matrix = rvec_matrix[0] # rodoriguesから抜き出し
        # 並進ベクトルの転置
        transpose_tvec = tvec[np.newaxis, :].T
        # 合成
        proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
        # オイラー角への変換
        euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
        euler_angle = np.squeeze(euler_angle)

        # print("x : " + str(tvec[0]))
        # print("y : " + str(tvec[1]))
        # print("z : " + str(tvec[2]))
        # print("roll : " + str(euler_angle[0]))
        # print("pitch: " + str(euler_angle[1]))
        # print("yaw  : " + str(euler_angle[2]))

        # print(tvec, euler_angle)                    

        # rvec = rvec[0, 0, :]
        # tvec = tvec[0, 0, :]
        # print(rvec.shape, rvec, tvec.shape, tvec)


        frame = aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 10)
        
        assert corner.shape == (1, 4, 2)
        pxy = corner[0,:,:].mean(axis=0)

        tvec[1] = - tvec[1]

        vecs[idx] = np.concatenate([tvec, pxy])

    return frame, vecs

if __name__ == '__main__':

    params = read_params()
    marker_ids = params['marker-ids']

    init_markers(params)

    initCamera()

    layout = [
        [ sg.Text('', key='-msg-')]
        ,
        [ sg.Button('Close') ]
    ]

    window = sg.Window('marker', layout, element_justification='c') # disable_minimize=True, 

    last_capture = time.time()
    while True:
        event, values = window.read(timeout=1)

        if event == sg.WIN_CLOSED or event == 'Close':

            closeCamera()
            break

        else:

            frame = getCameraFrame()

            frame, vecs = detect_markers(marker_ids, frame)

            msg = ''
            if 1 <= time.time() - last_capture:
                last_capture = time.time()
                window['-msg-'].update(value=msg)

            cv2.imshow("camera", frame)
