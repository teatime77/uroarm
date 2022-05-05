# 【python/OpenCV】カメラ映像をキャプチャするプログラム
# https://rikoubou.hatenablog.com/entry/2019/03/07/153430

import cv2
import time
import json
from datetime import datetime
import numpy as np
import pathlib
import shutil
import glob
import os

from util import getGlb

colab_dir = 'G:/マイドライブ/colab/ODTK/data/io'
detections_json = 'detections.json'

wait_infer = False
ann = None

CamX = 0
CamY = 0


def initCamera():
    global cap

    cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する

    print('BRIGHTNESS', cap.get(cv2.CAP_PROP_BRIGHTNESS))
    print('EXPOSURE'  , cap.get(cv2.CAP_PROP_EXPOSURE))
    print('FPS'       , cap.get(cv2.CAP_PROP_FPS))

    WIDTH = 960
    HEIGHT = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH ) == WIDTH
    assert cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == HEIGHT

def showMark(values, frame, h_lo, h_hi):
    try:
        # h_lo = float(values['-Hlo-'])
        # h_hi = float(values['-Hhi-'])

        s_lo = float(values['-Slo-'])

        v_lo = float(values['-Vlo-'])
    except:
        return (None, None)

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

    contours = [ x for x in contours if cv2.arcLength(x,True) < 150 ]
    if len(contours) != 0:
        areas = [ cv2.contourArea(cont) for cont in contours ]

        max_index = areas.index( max(areas) )
        cont = contours[max_index]

        cv2.drawContours(frame, contours, max_index, (255,255,255), -1)

        M = cv2.moments(cont)

        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # print(cv2.arcLength(cont,True))

            # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), thickness=-1)
            cv2.line(frame, (cx, cy - 10), (cx, cy + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_8)
            cv2.line(frame, (cx - 10, cy), (cx + 10, cy), (0, 0, 0), thickness=2, lineType=cv2.LINE_8)

            return (cx, cy)

    return (None, None)

def getCameraFrame():
    ret, frame = cap.read()

    sz = 720
    h, w, c = frame.shape
    assert(sz <= w and sz <= h)
    h1 = (h - sz) // 2
    h2 = h1 + sz

    w1 = (w - sz) // 2
    w2 = w1 + sz

    frame = frame[h1:h2, w1:w2, : ]

    return frame

def readCamera(values):
    global cap, CamX, CamY, ann

    frame = getCameraFrame()

    # 青玉の中心
    bx, by = showMark(values, frame, 10, 30)
    
    # 赤玉の中心
    rx, ry = showMark(values, frame, 80, 110)

    if bx is not None and rx is not None:
        CamX = 0.5 * (bx + rx)
        CamY = 0.5 * (by + ry)
    # print(bx, by, rx, ry, CamX, CamY)

    if wait_infer:
        getDetections()

    if ann is not None:
        showAnnotations(frame)
        ann = None

    cv2.imshow("camera", frame)

    return frame

def camX():
    return CamX

def camY():
    return CamY

def sendImage(values):
    global wait_infer, ann

    wait_infer = True
    ann = None

    frame = getCameraFrame()

    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = date + ".png"

    src_path  = f'./img/{file_name}'
    dst_path = f'{colab_dir}/img/{file_name}'

    cv2.imwrite(src_path, frame) # ファイル保存

    shutil.copy2(src_path, dst_path)
    print(src_path, '=>', dst_path)

    pathlib.Path(f'{colab_dir}/start').touch()
    
    print('start')

def getDetections():
    global wait_infer, ann

    # 推論が完了するのを待つ。
    end_file = f'{colab_dir}/end'

    if not os.path.exists(end_file):
        return

    os.remove(end_file)

    clearImage()

    wait_infer = False

    # 推論の結果を受け取る。
    shutil.copy2(f'{colab_dir}/{detections_json}', detections_json)

    with open(detections_json) as f:
        df = json.load(f)

    assert len(df['images']) == 1

    img_inf = df['images'][0]

    anns_id = [ x for x in df['annotations'] if x['image_id'] == img_inf['id'] ]
    max_score = max([ x['score'] for x in anns_id ])   

    ann = [ x for x in anns_id if x['score'] == max_score ][0]
    print('%s %d' % (img_inf['file_name'], 100 * max_score))

def showAnnotations(frame):
    
    if len(ann['bbox']) == 4:
        x, y, w, h = [ int(i) for i in ann['bbox']]
        frame = cv2.rectangle(frame,(x,y),(x + w,y + h),(0,255,0),3)

    else:
        x, y, w, h, theta = [ float(i) for i in ann['bbox']]

        glb = getGlb()
        glb.inferX = x + 0.5 * w
        glb.inferY = y + 0.5 * h

        minx, miny = [ glb.inferX - 0.5 * w, glb.inferY - 0.5 * h ]
        corners = np.array([ [ minx, miny ], [ minx + w, miny ], [ minx + w, miny + h ], [ minx, miny + h ]  ])
        centre = np.array([glb.inferX, glb.inferY])

        # cv2.rectangle(frame, np.int0(corners[0,:]), np.int0(corners[2,:]), (0,255,0),3)

        theta = - theta
        rotation = np.array([ [ np.cos(theta), -np.sin(theta) ],
                            [ np.sin(theta),  np.cos(theta) ] ])

        corners = np.matmul(corners - centre, rotation) + centre
        corners = np.int0(corners)
        cv2.drawContours(frame, [ corners ], 0, (255,0,0),2)

        cv2.circle(frame, (int(glb.inferX), int(glb.inferY)), 5, (255, 255, 255), thickness=-1)

        if glb.regX is None:
            print('infer:%.1f %.1f' % (glb.inferX, glb.inferY))
        else:
            prd_x = glb.regX.predict([[ glb.inferX, glb.inferY ]])
            prd_y = glb.regY.predict([[ glb.inferX, glb.inferY ]])

            glb.prdX = prd_x[0]
            glb.prdY = prd_y[0]

            print('infer:%.1f %.1f prd:%.1f %.1f ' % (glb.inferX, glb.inferY, glb.prdX, glb.prdY))

def clearImage():
    for img_path in glob.glob(f'{colab_dir}/img/*.png'):
        os.remove(img_path)
        print('rm', img_path)

    for img_path in glob.glob(f'./img/*.png'):
        os.remove(img_path)
        print('rm', img_path)


def closeCamera():
    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()