import os
import queue
from multiprocessing import Process, Queue, freeze_support
import cv2
import numpy as np
from openvino.runtime import Core
from openvino.pyopenvino import ConstOutput
import time
from util import read_params, download_file
from camera import initCamera, getCameraFrame, closeCamera

def getInputImg(bmp):

    img = bmp.astype(np.float32)
    img = img / 255.0

    img = cv2.resize(img, dsize=(1280,1280))

    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :]

    return img

def get_center_of_max_score_box(scores, boxes, bmp_shape):
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

    bmp_h = bmp_shape[0]
    bmp_w = bmp_shape[1]

    num_box_h = box.shape[2]
    num_box_w = box.shape[3]

    box_h = bmp_h / num_box_h
    box_w = bmp_w / num_box_w

    center_y = int(row * box_h + box_h / 2) 
    center_x = int(col * box_w + box_w / 2)

    return center_x, center_y

def infer_img(img_que : Queue, result_que : Queue):
    ie = Core()

    devices = ie.available_devices

    for device in devices:
        device_name = ie.get_property(device_name=device, name="FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")

    model_name = 'ichigo_16'

    model_bin_file = f'data/model/{model_name}.bin'
    if not os.path.isfile(model_bin_file):

        print('downloading model file...')

        url = f'http://lang.main.jp/uroa-data/model/{model_name}.bin'
        download_file(url, model_bin_file)

        print('download completed.')


    model = ie.read_model(model=f'data/model/{model_name}.xml')
    compiled_model = ie.compile_model(model=model, device_name="GPU")

    input_layer_ir = next(iter(compiled_model.inputs))

    # Create inference request
    request = compiled_model.create_infer_request()

    while True:
        bmp = img_que.get()
        if bmp is None:
            result_que.put((np.nan, np.nan))
            break

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

        center_x, center_y = get_center_of_max_score_box(scores, boxes, bmp.shape)

        result_que.put((center_x, center_y))

class Inference():
    def __init__(self):
        self.img_que = Queue()
        self.result_que = Queue()
        
        self.cx = np.nan
        self.cy = np.nan
        
        self.pw = Process(target=infer_img, args=(self.img_que, self.result_que))

        self.pw.start()
    
    def get(self, frame):
        h, w = frame.shape[:2]
        assert h == w

        if self.img_que.empty():

            if h == 720:
                frame2 = frame
            else:
                frame2 = cv2.resize(frame, (720, 720))

            self.img_que.put(frame2)

        try:
            cx, cy = self.result_que.get_nowait()

            assert not np.isnan(cx) and not np.isnan(cy)

            cx = cx * w // 720
            cy = cy * h // 720

            self.cx = cx
            self.cy = cy

        except queue.Empty:
            pass

        return self.cx, self.cy

    def close(self):
        self.img_que.put(None)

        while True:
            cx, cy = self.result_que.get()
            if np.isnan(cx):
                break


if __name__ == '__main__':
    freeze_support()

    params = read_params()
    initCamera(params)

    cv2.namedWindow('window')

    inference = Inference()


    cx, cy = (None, None)
    while True:
        frame = getCameraFrame()

        cx, cy = inference.get(frame)

        if not np.isnan(cx):
            cv2.circle(frame, (int(cx), int(cy)), 10, (255,0,0), -1)


        h, w = frame.shape[:2]
        assert h == w
        if h <= 720:
            frame2 = frame
        else:
            frame2 = cv2.resize(frame, (720, 720))

        cv2.imshow('window', frame2)

        k = cv2.waitKey(1)
        if k == ord('q'):
            inference.close()
            closeCamera()
            break

