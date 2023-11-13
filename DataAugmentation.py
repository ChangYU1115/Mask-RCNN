import os
from pickletools import uint8
import numpy as np
import cv2

def dataaugment(original_datasets_path, new_datasets_path, mode = 0,visualize = False, scale = 1.0):
    IsExist(new_datasets_path)
    Images_path = original_datasets_path + "Images/"
    Masks_path = original_datasets_path + "Masks/"
    new_Images_path = new_datasets_path + "Images/"
    new_Masks_path = new_datasets_path + "Masks/"

    filenum = countdir(Images_path)
    IsExist(new_Images_path)
    IsExist(new_Masks_path)
    for idx, filename in enumerate(os.listdir(Masks_path)):
        print(f"{Images_path + filename} augment...")
        img = cv2.imread(Images_path  + filename + ".png")
        mask = cv2.imread(Masks_path + filename + "/0.png")

        mask = np.array(mask)
        classes = np.unique(mask)[1]
        masks = mask == classes

        xnmin, xnmax, ynmin, ynmax = anchor_generation(masks, scale = scale)
        nimg = img[ynmin:ynmax, xnmin:xnmax]
        nmask = mask[ynmin:ynmax, xnmin:xnmax]
        
        nimg = cv2.resize(nimg, (512,512), interpolation = cv2.INTER_NEAREST)
        nmask = cv2.resize(nmask, (512,512), interpolation = cv2.INTER_NEAREST)

        nid = filenum + idx

        output_Masks_path = new_Masks_path + str(nid) + "/"
        
        output_Masks_path = output_Masks_path + "0.png"
        outout_Images_path = new_Images_path + str(nid) + ".png"

        if visualize == True:
            mask = ((mask == classes)*255).astype(np.uint8)
            nmask = ((nmask == classes)*255).astype(np.uint8)

        if mode == 0:
            IsExist(new_Masks_path + filename)
            IsExist(output_Masks_path)
            cv2.imwrite(new_Images_path + filename + ".png", img)
            cv2.imwrite(new_Masks_path + filename + "/0.png", mask)
            cv2.imwrite(outout_Images_path, nimg)
            cv2.imwrite(output_Masks_path, nmask)

        if mode == 1:
            IsExist(new_Masks_path + filename)        
            cv2.imwrite(new_Images_path + filename + ".png", nimg)
            cv2.imwrite(new_Masks_path + filename + "/0.png", nmask)


def anchor_generation(masks, scale):
    length = 512

    pos = np.where(masks)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])

    xnmin = int(xmin*scale)
    xnmax = xmax + int((length - xmax)*(1 - scale))
    ynmin = int(ymin*scale)
    ynmax = ymax + int((length - ymax)*(1 - scale))
    # xmin, xmax, ymin, ymax
    # xnmin, xnmax, ynmin, ynmax
    return xnmin, xnmax, ynmin, ynmax



def countdir(dir):
    initial_count = 0
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            initial_count += 1
    return initial_count
    
def IsExist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

dataaugment("./TrainMask/","./DataAug1/", mode = 1, visualize = True, scale = 0.2)
