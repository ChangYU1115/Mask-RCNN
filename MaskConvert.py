import os
import cv2

def convert(input_path = "./original/", output_path = "./Convert/"):
    DirIsExist(output_path)
    DirIsExist(output_path + "Images/")
    DirIsExist(output_path + "Masks/")

    classes = [i for i in os.listdir(input_path)]
    
    idx = 1
    txt = ""
    for class_name in classes:
        img_path = input_path + class_name + "/Full/"
        mask_path = input_path + class_name + "/ROI/"
        txt = txt + class_name + " " + str(idx).zfill(4) + "-"
        for img_name in os.listdir(img_path):
            # I/O path
            img_input = img_path + img_name
            img_output = output_path + "Images/" + str(idx).zfill(4) + ".png"
            mask_input = mask_path + img_name.split('.')[0] + "_mask.png"
            mask_output = output_path + "Masks/" + str(idx).zfill(4) + ".png"
            # read
            img = cv2.imread(img_input)
            mask = cv2.imread(mask_input)
            # write
            cv2.imwrite(img_output, img)
            cv2.imwrite(mask_output, mask)
            
            idx = idx + 1
        txt = txt + str(idx-1).zfill(4) + "\n"
    txt = txt[:-1]
    f = open(output_path + "label.txt", "w")
    f.write(txt)
    f.close()

def DirIsExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

convert(input_path = "./original/", output_path = "./Convert/")
