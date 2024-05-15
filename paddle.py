#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import cv2
from datetime import datetime, timedelta
from ppocr_onnx.ppocr_onnx import PaddleOcrONNX
import os
import time
import numpy as np






def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", type=str, default='./ppocr_onnx/imgs/1.png')

    parser.add_argument(
        "--det_model",
        type=str,
        default='./ppocr_onnx/model/det_model/en_PP-OCRv3_det_infer.onnx',
    )
    parser.add_argument(
        "--rec_model",
        type=str,
        default='./ppocr_onnx/model/rec_model/en_PP-OCRv3_rec_infer.onnx',
    )
    parser.add_argument(
        "--rec_char_dict",
        type=str,
        default='./ppocr_onnx/ppocr/utils/dict/en_dict.txt',
    )
    parser.add_argument(
        "--cls_model",
        type=str,
        default=
        './ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    )

    parser.add_argument(
        "--use_gpu",
        action="store_true",
    )

    args = parser.parse_args()

    return args


class DictDotNotation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self




def get_paddleocr_parameter():
    paddleocr_parameter = DictDotNotation()

    # params for prediction engine
    paddleocr_parameter.use_gpu = False

    # params for text detector
    paddleocr_parameter.det_algorithm = 'DB'
    paddleocr_parameter.det_model_dir = './ppocr_onnx/model/det_model/ch_PP-OCRv3_det_infer.onnx'
    paddleocr_parameter.det_limit_side_len = 960
    paddleocr_parameter.det_limit_type = 'max'
    paddleocr_parameter.det_box_type = 'quad'

    # DB parmas
    paddleocr_parameter.det_db_thresh = 0.3
    paddleocr_parameter.det_db_box_thresh = 0.6
    paddleocr_parameter.det_db_unclip_ratio = 1.5
    paddleocr_parameter.max_batch_size = 10
    paddleocr_parameter.use_dilation = False
    paddleocr_parameter.det_db_score_mode = 'fast'

    # params for text recognizer
    paddleocr_parameter.rec_algorithm = 'SVTR_LCNet'
    paddleocr_parameter.rec_model_dir = './ppocr_onnx/model/rec_model/japan_PP-OCRv3_rec_infer.onnx'
    paddleocr_parameter.rec_image_shape = '3, 48, 320'
    paddleocr_parameter.rec_batch_num = 6
    paddleocr_parameter.rec_char_dict_path = './ppocr_onnx/ppocr/utils/dict/japan_dict.txt'
    paddleocr_parameter.use_space_char = True
    paddleocr_parameter.drop_score = 0.5

    # params for text classifier
    paddleocr_parameter.use_angle_cls = False
    paddleocr_parameter.cls_model_dir = './ppocr_onnx/model/cls_model/ch_ppocr_mobile_v2.0_cls_infer.onnx'
    paddleocr_parameter.cls_image_shape = '3, 48, 192'
    paddleocr_parameter.label_list = ['0', '180']
    paddleocr_parameter.cls_batch_num = 6
    paddleocr_parameter.cls_thresh = 0.9

    paddleocr_parameter.save_crop_res = False

    return paddleocr_parameter


def main():
    # コマンドライン引数
    args = get_args()
    # PaddleOCR準備
    paddleocr_parameter = get_paddleocr_parameter()

    paddleocr_parameter.det_model_dir = args.det_model
    paddleocr_parameter.rec_model_dir = args.rec_model
    paddleocr_parameter.rec_char_dict_path = args.rec_char_dict
    paddleocr_parameter.cls_model_dir = args.cls_model

    paddleocr_parameter.use_gpu = args.use_gpu

    paddle_ocr_onnx = PaddleOcrONNX(paddleocr_parameter)

    # Show camera and perform OCR
    show_camera()

    # OCR実施
    #dt_boxes, rec_res, time_dict = paddle_ocr_onnx(image)
    # os.remove(image_path)  # Not sure why you're removing an image path here
    # print(time_dict)
    #for dt_box, rec in zip(dt_boxes, rec_res):
        # print(dt_box, rec)
        # print(rec)
        #print("---------------------------------------------------------------")
        #print(rec[0])  # Make sure to indent this line properly







def gstreamer_pipeline(
    sensor_id=0,
    capture_width=540,
    capture_height=960,
    display_width=540,
    display_height=960,
    framerate=30,
    flip_method=0,
    format="NV12",
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,

        )
    )




def show_camera():
    window_title = "CSI Camera"

    args = get_args()
    paddleocr_parameter = get_paddleocr_parameter()

    paddleocr_parameter.det_model_dir = args.det_model
    paddleocr_parameter.rec_model_dir = args.rec_model
    paddleocr_parameter.rec_char_dict_path = args.rec_char_dict
    paddleocr_parameter.cls_model_dir = args.cls_model

    paddleocr_parameter.use_gpu = args.use_gpu

    paddle_ocr_onnx = PaddleOcrONNX(paddleocr_parameter)

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                  #  time.sleep(10)  
                   # Inside the show_camera() function

# After capturing a frame, print its dimensions
                    frame_uint32 = frame.astype(np.uint32)
                    print("Frame dimensions:", frame.shape[1], "x", frame.shape[0])  # Width x Height
                    print("Data type of points:", frame_uint32.dtype)
                    dt_boxes, rec_res, time_dict = paddle_ocr_onnx(frame_uint32)
                   # time.sleep(5)  # Delay OCR processing for 6 seconds
                    # print(time_dict)
                    for dt_box, rec in zip(dt_boxes, rec_res):
                        # print(dt_box, rec)
                        # print(rec)
                        print("---------------------------------------------------------------")
                        print(rec[0])  # Make sure to indent this line properly
                else:
                    break 
                    
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


















if __name__ == '__main__':
    start_time = datetime.now()

    main()
    end_time = datetime.now()
    time_difference = end_time - start_time
    print("=============================================")
    print("Time Difference:", time_difference)
