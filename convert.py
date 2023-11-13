from msilib.schema import Binary
import os
import cv2
import json
import numpy as np

'''
datas keys = ['filename']
    data keys = ['filename', 'size', 'regions', 'file_attributes']
        filename = Image file name

        size = A number

        region keys = ['shape_attributes', 'region_attributes']
            region_attributes keys = ['names']
                names = class1, class2 ...
            shape_attributes keys = ['name', 'all_points_x', 'all_points_y']
                name keys = polyline
                area = ['all_points_x', 'all_points_y']
        
        file_attributes = {}
'''

def DirIsExist(path):
    os.makedirs(path) if not os.path.exists(path) else print(f"{path} exist!")

def mask2json(inputpath, outputpath, classes = ['benign','malignant']):
    inputpath = "./TestMask/"
    outputpath = "./TestJSON/"

    Jfiledata = {}
    jsonpath = outputpath + outputpath.split("/")[-2] + "_json.json"
    Images_path = inputpath + "Images/"
    Masks_path = inputpath+ "Masks/"

    for filename in os.listdir(Images_path):
        img = cv2.imread(Images_path + filename)
        cv2.imwrite(outputpath + filename, img)
        Ms = []
        for maskfilename in os.listdir(Masks_path + filename.split(".")[0] + "/"):
            M = {}
            
            Mask = cv2.imread(Masks_path + filename.split(".")[0] + "/" + maskfilename)
            Binary_Mask = ((Mask[:,:,0] > 0)*255).astype(np.uint8)
            contours, hirtatchy = cv2.findContours(Binary_Mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            X = []
            Y = []
            area = int(cv2.contourArea(contours[0]))
            for point in contours[0]:
                x, y = point[0]
                X.append(int(x))
                Y.append(int(y))

            C = classes[np.unique(Mask)[1] - 1]
            M.update({'shape_attributes':{'name': 'polyline', 'all_points_x': X, 'all_points_y': Y}})
            M.update({'region_attributes':{'names':C}})

            Ms.append(M)
        
        Jfiledata.update({filename+str(area):{'filename':filename, 'size':area, 'regions':Ms, 'file_attributes':{}}})

    with open(jsonpath, "w") as write_file:
        json.dump(Jfiledata, write_file, indent=4)



def json2mask(inputpath, outputpath, classes = ['benign','malignant']):
    # inputpath = "./val/"
    jsonpath = inputpath + inputpath.split("/")[-2] + "_json.json"
    # outputpath = "./TestMask/"

    jfile = open(jsonpath, 'r')
    datas = json.load(jfile)
    jfile.close()

    Images_path = outputpath + "Images/"
    Masks_path = outputpath + "Masks/"

    DirIsExist(Images_path)
    DirIsExist(Masks_path)

    file_ids = [key for key in datas.keys()]
    for file_id in file_ids:
        filename, size, regions, file_attributes = datas[file_id]['filename'], datas[file_id]['size'], datas[file_id]['regions'], datas[file_id]['file_attributes']

        img = cv2.imread(inputpath + filename)
        cv2.imwrite(Images_path + filename.split(".")[0] + ".png", img)

        GTDIR = filename.split('.')[0]
        DirIsExist(Masks_path + GTDIR + "/")
        for idx, region in enumerate(regions):
            region_class = classes.index(region['region_attributes']['names'])

            area = [[x, y] for (x, y) in zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y'])]
            filled = np.zeros_like(img[:,:,0]).astype(np.uint8)
            filled = cv2.fillPoly(filled, pts = [np.array(area)], color =(region_class + 1)).astype(np.uint8)
            cv2.imwrite(f"{Masks_path}{GTDIR}/{idx}.png", filled)



json2mask("./val/", "./TestMask/", classes = ['benign','malignant'])