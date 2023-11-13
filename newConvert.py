import os
import cv2

def convert(input_path = "./original/", output_path = "./Convert/"):
    DirIsExist(output_path)
    DirIsExist(output_path + "Images/")
    DirIsExist(output_path + "Masks/")

    classes = ['benign', 'malignant']
    
    idx = 1
    txt = ""
    for ob_id, class_name in enumerate(classes):
        img_path = os.path.join(input_path, class_name)
        mask_path = os.path.join(input_path, class_name)
        # txt = txt + class_name + " " + str(idx).zfill(4) + "-"
        for img_name in os.listdir(img_path):
            # I/O path
            if img_name.endswith(').png'):
                img_input = os.path.join(img_path, img_name)
                img_output = output_path + "/Images/" + str(idx).zfill(4) + ".png"
                mask_input = os.path.join(mask_path, img_name.split('.')[0] + "_mask.png")
                DirIsExist(output_path + "/Masks/" + str(idx).zfill(4))
                mask_output = output_path + "Masks/" + str(idx).zfill(4) + "/0.png"
                print(img_input, mask_input)
                # read
                img = cv2.imread(img_input)
                mask = (cv2.imread(mask_input)/255)*(ob_id + 1)

                # write
                cv2.imwrite(img_output, img)
                cv2.imwrite(mask_output, mask)

                idx = idx + 1
        # txt = txt + str(idx-1).zfill(4) + "\n"
    # txt = txt[:-1]
    # f = open(output_path + "label.txt", "w")
    # f.write(txt)
    # f.close()

def DirIsExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

convert(input_path = "./TestData/", output_path = "./newTest/")
