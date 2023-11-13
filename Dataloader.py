import os
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random
from torchvision import transforms as TF
from torchvision.transforms import InterpolationMode
from transforms import RandomHorizontalFlip

# , T = False
class DatasetReader(torch.utils.data.Dataset):
    def __init__(self, root, transforms, T = True, train = True, mode = 0):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "Images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "Masks"))))

        self.mode1 = TF.Compose([
                TF.RandomCrop(256),
                TF.Resize(512, interpolation=InterpolationMode.NEAREST, max_size=None, antialias=None)
            ])

        if T == True:
            self.TransControl = True
            self.TransF = TF.Compose([
                TF.RandomVerticalFlip(0.5),
                TF.RandomHorizontalFlip(0.5),
                TF.RandomRotation(5)
            ])
        else:
            self.TransControl = False
        
        self.mode = mode
        self.train = train

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "Images", self.imgs[idx])
        mask_paths = os.path.join(self.root, "Masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask_path = random.choice([mask_path for mask_path in os.listdir(mask_paths)])
        mask = Image.open(mask_paths + "/" + mask_path)



        if self.train:
            if np.shape(mask) == (512,512):
                mask = np.array(mask)
                mask = cv2.merge([mask, mask, mask])
                mask = Image.fromarray(mask)

        if (not self.train) and np.shape(mask) == (512, 512, 3):
                mask = np.array(mask)
                mask = mask[:,:,0]
                mask = Image.fromarray(mask)

        if self.train:
            if self.mode == 1:
                if randomtoss():
                    seed = random.randint(0, 2**32)
                    torch.manual_seed(seed)
                    nimg = self.mode1(img)
                    torch.manual_seed(seed)
                    nmask = self.mode1(mask)

                    if 1 in np.unique(nmask) | 2 in np.unique(nmask):
                        img = nimg
                        mask = nmask
                
            elif self.mode == 2:
                if randomtoss():
                    img = np.array(img)
                    mask = np.array(mask)
                    mask, img = Augmentation(img, mask, scale = 0.1)
                    img = Image.fromarray(img)
                    mask = Image.fromarray(mask)

        
        if self.TransControl == True:
            seed = random.randint(0, 2**32)
            torch.manual_seed(seed)
            img = self.TransF(img)
            torch.manual_seed(seed)
            mask = self.TransF(mask)

        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        if num_objs >= 1:
            masks = mask == obj_ids[:, None, None]
        else:
            masks = mask.copy()
        # get bounding box coordinates for each mask

#----------------------------------------------------------------------------------------
        image_id = torch.tensor([idx])
        if num_objs >= 1:
            labels = torch.ones((num_objs,), dtype=torch.int64)*obj_ids[0]
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        else:
            labels = torch.zeros((1,), dtype=torch.int64)
            iscrowd = torch.zeros((1,), dtype=torch.int64)
        
        boxes = []
        if num_objs >= 1:
            for i in range(num_objs):
                if self.train:
                    pos = np.where(masks)
                else:
                    pos = np.where(masks[0])

                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            
            if self.train:
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                masks = torch.unsqueeze(masks, 0)[:,:,:,0]

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            boxes.append([0, 0, 512, 512])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            if self.train:
                masks = torch.as_tensor(masks, dtype=torch.uint8)
                masks = torch.unsqueeze(masks, 0)[:,:,:,0]
#----------------------------------------------------------------------------------------
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)


        return img, target

    def __len__(self):
        return len(self.imgs)

def Augmentation(img, mask, scale = 0.1):
    mask = np.array(mask)
    classes = np.unique(mask)[1]
    masks = mask == classes

    xnmin, xnmax, ynmin, ynmax = anchor_generation(masks, scale = scale)
    nimg = img[ynmin:ynmax, xnmin:xnmax]
    nmask = mask[ynmin:ynmax, xnmin:xnmax]

    nimg = cv2.resize(nimg, (512,512), interpolation = cv2.INTER_NEAREST)
    nmask = cv2.resize(nmask, (512,512), interpolation = cv2.INTER_NEAREST)
    return nmask, nimg

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

def randomtoss(r = 0.5):
    if random.random() >= r:
        return True
    else:
        return False