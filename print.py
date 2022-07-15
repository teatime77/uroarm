import os
import numpy as np
from cv2 import aruco
import cv2

# マーカーの保存先
dir_marker = 'data/marker'

def make_board():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    squares_x = 5
    squares_y = 7
    square_length_meter = 0.04
    marker_length_meter = 0.02
    charucoBoard = aruco.CharucoBoard_create(squares_x, squares_y, square_length_meter, marker_length_meter, dictionary)

    square_length_mm = 1000 * square_length_meter

    # mm to inch
    mm_to_inch = 1 / 25.4

    # dots per inch
    dpi = 350

    image_x_pixel = round(squares_x * square_length_mm * mm_to_inch * dpi)
    image_y_pixel = round(squares_y * square_length_mm * mm_to_inch * dpi)
    
    image = charucoBoard.draw((image_x_pixel, image_y_pixel))

    file_path = f'{dir_marker}/board4x4.png'
    cv2.imwrite(file_path, image)

    print(f'board image file is created: {file_path}')

def make_markers():
    ### --- parameter --- ###

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

    file_path = f'{dir_marker}/markers4x4.png'
    cv2.imwrite(file_path, bg)     

    print(f'markers image file is created: {file_path}')

if __name__ == '__main__':

    os.makedirs(dir_marker, exist_ok=True)

    make_board()
    make_markers()