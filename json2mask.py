import matplotlib.pyplot as plt
# import matplotlib.image as img
import json
import numpy as np
import cv2
from PIL import Image
import torch
jfile = open("./val/val_json.json",'r')
datas = json.load(jfile)
jfile.close()
for data in datas:
    img = cv2.imread("./val/" + datas[data]['filename'])
    filled = np.zeros_like(img[:,:,0]).astype(np.float32)
    area = []
    for i in range(len(datas[data]['regions'][0]['shape_attributes']['all_points_x'])):
        area.append([datas[data]['regions'][0]['shape_attributes']['all_points_x'][i], datas[data]['regions'][0]['shape_attributes']['all_points_y'][i]])
    area = np.array(area)
    filled = cv2.fillPoly(filled, pts = [area], color =(255)).astype(np.uint8)
    if datas[data]['regions'][0]['region_attributes']['names'] == 'benign':
        blur = cv2.GaussianBlur(filled, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
        ret, th1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    elif datas[data]['regions'][0]['region_attributes']['names'] == 'malignant':
        blur = cv2.GaussianBlur(filled, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
        ret, th1 = cv2.threshold(blur, 127, 200, cv2.THRESH_BINARY)
    else:
        print("error")

    cv2.imwrite("./Test_Datasets/Mask/" + datas[data]['filename'].split('.')[0] + '.png' , th1)
    cv2.imwrite("./Test_Datasets/Images/" + datas[data]['filename'].split('.')[0] + '.png' , img)