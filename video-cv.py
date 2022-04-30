# 【python/OpenCV】カメラ映像をキャプチャするプログラム
# https://rikoubou.hatenablog.com/entry/2019/03/07/153430

import cv2
from datetime import datetime
import numpy as np
import sys
import pickle
import struct ### new code

if True:
    cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する

    WIDTH = 960
    HEIGHT = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    while True:
        ret, frame = cap.read()

        sz = 720
        h, w, c = frame.shape
        assert(sz <= w and sz <= h)
        h1 = (h - sz) // 2
        h2 = h1 + sz

        w1 = (w - sz) // 2
        w2 = w1 + sz

        frame = frame[h1:h2, w1:w2, : ]

        cv2.imshow("camera", frame)

        k = cv2.waitKey(1)&0xff # キー入力を待つ
        if k == ord('p'):
            # 「p」キーで画像を保存
            date = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = "./img/" + date + ".png"
            cv2.imwrite(path, frame) # ファイル保存
            print(path)

            # cv2.imshow(path, frame) # キャプチャした画像を表示
        elif k == ord('q'):
            # 「q」キーが押されたら終了する
            break

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()    

else:
    cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する

    # cap.set(cv2.CAP_PROP_FPS, FPS)

    # フォーマット・解像度・FPSの取得
    # fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    print("%dx%d fps:%d" % (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)))

    for cnt in range(10000):
        ret, frame = cap.read()
        # frame = np.array(frame)

        # if cnt % 10 == 0:
        cv2.imshow("camera", frame)

        # print(cnt)

    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()