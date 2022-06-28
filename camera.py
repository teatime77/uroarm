# 【python/OpenCV】カメラ映像をキャプチャするプログラム
# https://rikoubou.hatenablog.com/entry/2019/03/07/153430

import cv2
import numpy as np

from util import getGlb

colab_dir = 'G:/マイドライブ/colab/ODTK/data/io'
detections_json = 'detections.json'

wait_infer = False
ann = None

CamX = 0
CamY = 0


def initCamera():
    global cap, WIDTH, HEIGHT

    cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する

    print('BRIGHTNESS', cap.get(cv2.CAP_PROP_BRIGHTNESS))
    print('EXPOSURE'  , cap.get(cv2.CAP_PROP_EXPOSURE))
    print('FPS'       , cap.get(cv2.CAP_PROP_FPS))

    WIDTH = 960
    HEIGHT = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    w = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH ) )
    h = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    if w != WIDTH or h != HEIGHT:

        print(f'width:{w} height:{h} ==================================================')

        WIDTH  = w
        HEIGHT = h


def getCameraFrame():
    ret, frame = cap.read()

    sz = min(WIDTH, HEIGHT)
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

    if ann is not None:
        showAnnotations(frame)
        # ann = None

    cv2.imshow("camera", frame)

    return frame

def camX():
    return CamX

def camY():
    return CamY



def sendImage(values, inference):
    global wait_infer, ann

    # wait_infer = True
    ann = None

    frame = getCameraFrame()

    cx, cy = inference.get(frame)

    print('rcv', cx, cy)

    ann = {
        'bbox': [ cx, cy ]
    }

    return cx, cy

def Eye2Hand(eye_x, eye_y):
    glb = getGlb()

    prd_x = glb.regX.predict([[ eye_x, eye_y ]])
    prd_y = glb.regY.predict([[ eye_x, eye_y ]])

    hand_x = prd_x[0]
    hand_y = prd_y[0]

    return hand_x, hand_y

def showAnnotations(frame):
    if len(ann['bbox']) == 2:
        cx, cy = ann['bbox']

        cv2.circle(frame, (int(cx), int(cy)), 10, (255,255,255), -1)

    elif len(ann['bbox']) == 4:
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


def closeCamera():
    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()
