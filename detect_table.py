import cv2
import numpy as np
import os
from encopy import main as onnx_main

def table_detection(img):
    #img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)

    kernel_length_v = (np.array(img_gray).shape[1]) // 120
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length_v))
    im_temp1 = cv2.erode(img_bin, vertical_kernel, iterations=3)
    vertical_lines_img = cv2.dilate(im_temp1, vertical_kernel, iterations=3)

    kernel_length_h = (np.array(img_gray).shape[1]) // 40
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length_h, 1))
    im_temp2 = cv2.erode(img_bin, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(im_temp2, horizontal_kernel, iterations=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    table_segment = cv2.addWeighted(vertical_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    table_segment = cv2.erode(cv2.bitwise_not(table_segment), kernel, iterations=2)
    thresh, table_segment = cv2.threshold(table_segment, 0, 255, cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(table_segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    full_list = []
    row = []
    data1 = []
    first_iter = 0
    firsty = -1	
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        #print(x,y,w,h)
        # cropped = img[y:y + h, x:x + w]
        # cv2_imshow(cropped)
        # bounds = reader.readtext(cropped)
        # print(bounds)
        # print(x,y,w,h)

        if h > 9 and h < 100:
            if first_iter == 0:
                first_iter = 1
                firsty = y
            if firsty != y:
                row.reverse()
                full_list.append(row)
                row = []
                data1 = []
            #print(x, y, w, h)
            cropped = img[y:y + h, x:x + w]
            #cv2_imshow(cropped)
            data = onnx_main(cropped)
            output_data=""
            for i in data['text']:
                output_data=output_data+i
            #print(bounds)
            #print("++++++++++++++++++")

            try:
                data1.append(output_data)
                data1.append(w)
                row.append(data1)
                data1 = []
            except:
                data1.append("--")
                data1.append(w)
                row.append(data1)
                data1 = []
            firsty = y
    full_list.reverse()
    #print(full_list)

    new_data = []
    new_row = []
    for i in full_list:
        for j in i:
            #print(j)
            new_row.append(j[0])
        new_data.append(new_row)
        new_row = []
    #print(new_data)

    return new_data

