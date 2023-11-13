import os
import cv2
import torch
import numpy as np

import torch
from utils import get_model_instance_segmentation, get_transform

import utils
from Dataloader import DatasetReader
from engine import train_one_epoch, evaluate

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 3

    dataset_test = DatasetReader('./TestMask/', get_transform(train=False))

    indices_test = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices_test[:])
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1,collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes, train_pth = './model/best.pth')
    model.to(device)

    evaluate(model, data_loader_test, device=device, mask_predict = True)

    
