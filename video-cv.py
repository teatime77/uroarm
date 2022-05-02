# 【python/OpenCV】カメラ映像をキャプチャするプログラム
# https://rikoubou.hatenablog.com/entry/2019/03/07/153430

import cv2
from datetime import datetime
import numpy as np
import pathlib
import shutil
import glob
import os

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

def readCamera():
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

def sss():
    colab_dir = 'G:/マイドライブ/colab/ODTK/data/io'

    k = cv2.waitKey(1)&0xff # キー入力を待つ
    if k == ord('p'):
        # 「p」キーで画像を保存
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = date + ".png"
        src_path  = f'./img/{file_name}'
        dst_path = f'{colab_dir}/img/{file_name}'
        cv2.imwrite(src_path, frame) # ファイル保存
        shutil.copy2(src_path, dst_path)
        print(src_path, '=>', dst_path)

        # cv2.imshow(path, frame) # キャプチャした画像を表示
    elif k == ord('s'):
            
        pathlib.Path(f'{colab_dir}/start').touch()
        print('start')

    elif k == ord('e'):
        for img_path in glob.glob(f'{colab_dir}/img/*.png'):
            os.remove(img_path)

        for img_path in glob.glob(f'./img/*.png'):
            os.remove(img_path)

        print('erase')

    elif k == ord('q'):
        # 「q」キーが押されたら終了する
        pass

def closeCamera():
    # キャプチャをリリースして、ウィンドウをすべて閉じる
    cap.release()
    cv2.destroyAllWindows()    





# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()