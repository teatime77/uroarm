# 【python/OpenCV】カメラ映像をキャプチャするプログラム
# https://rikoubou.hatenablog.com/entry/2019/03/07/153430

import cv2

def initCamera(params):
    global cap, WIDTH, HEIGHT

    cap = cv2.VideoCapture(params['camera-index'])

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

def closeCamera():
    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()
