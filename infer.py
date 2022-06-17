import sys
import queue
import glob
from multiprocessing import Process, Queue, freeze_support
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import openvino
from openvino.runtime import Core
from openvino.pyopenvino import ConstOutput
import time

def getInputImg(bmp):

    img = bmp.astype(np.float32)
    img = img / 255.0

    img = cv2.resize(img, dsize=(1280,1280))

    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]

    return img

def getBox(scores, boxes, shape):
    max_score = 0
    max_score_idx = 0
    for score_idx, score in enumerate(scores):
        idx = np.unravel_index(score.argmax(), score.shape)
        if max_score < score[idx]:
            max_score = score[idx]
            max_score_idx = score_idx
            max_idx = idx
    b, a, row, col = max_idx
    a1 = a * 6
    a2 = (a + 1) * 6
    score = scores[max_score_idx]
    box = boxes[max_score_idx]
    print("    ", scores[max_score_idx].shape, box.shape, max_idx, max_score, score[max_idx], box[0, a1:a2, row, col])

    bmp_h = shape[0]
    bmp_w = shape[1]

    num_box_h = box.shape[2]
    num_box_w = box.shape[3]

    box_h = bmp_h / num_box_h
    box_w = bmp_w / num_box_w

    cy = int(row * box_h + box_h / 2) 
    cx = int(col * box_w + box_w / 2)

    return cx, cy

def infer_img(img_que : Queue, result_que : Queue):
    ie = Core()

    devices = ie.available_devices

    for device in devices:
        device_name = ie.get_property(device_name=device, name="FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")

    model_file = 'model.xml'
    model_file = 'cola_720.xml'
    model_file = 'cola_16.xml'
    model_file = 'ichigo_16.xml'

    img_dir = '../data/cola/img'
    img_dir = '../data/ichigo/img'

    model = ie.read_model(model=model_file)
    compiled_model = ie.compile_model(model=model, device_name="GPU")

    input_layer_ir = next(iter(compiled_model.inputs))

    # Create inference request
    request = compiled_model.create_infer_request()

    while True:
        bmp = img_que.get()

        img = getInputImg(bmp)

        print('start infer')

        start_time = time.time()
        ret = request.infer({input_layer_ir.any_name: img})
        sec = '%.1f' % (time.time() - start_time)
        
        print(sec, type(ret))

        scores = [None] * 5
        boxes  = [None] * 5

        score_names = [ 'score_1', 'score_2', 'score_3', 'score_4', 'score_5' ]
        box_names = [ 'box_1', 'box_2', 'box_3', 'box_4', 'box_5' ]
        for k, v in ret.items():
            assert type(k) is ConstOutput
            print("    ", type(k), type(k.names), k.names, type(v), v.shape)

            name = k.any_name
            if name in score_names:
                score_idx =  score_names.index(name)
                scores[score_idx] = v

            if name in box_names:
                box_idx = box_names.index(name)
                boxes[box_idx] = v

        cx, cy = getBox(scores, boxes, bmp.shape)

        result_que.put((cx, cy))


if __name__ == '__main__':
    freeze_support()


    cv2.namedWindow('window')

    cap = cv2.VideoCapture(0) # 任意のカメラ番号に変更する

    img_que = Queue()
    result_que = Queue()

    pw = Process(target=infer_img, args=(img_que, result_que))

    pw.start()

    ret, bmp = cap.read()
    img_que.put(bmp.copy())

    cx, cy = (None, None)
    while True:
        ret, bmp = cap.read()

        try:
            cx, cy = result_que.get_nowait()
            img_que.put(bmp.copy())

        except queue.Empty:
            pass

        if cx is not None:

            cv2.circle(bmp, (int(cx), int(cy)), 10, (255,255,255), -1)

        cv2.imshow('window', bmp)
        k = cv2.waitKey(1)
        # if k == ord('q'):
        #     break
