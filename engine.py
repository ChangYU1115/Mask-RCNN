import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

import os 
import cv2
import numpy as np


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    F = open("Loss.txt", "a")
    txt = ""
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        for loss_key in ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']:
            txt = txt + f"{round(loss_dict[loss_key].item(), 3)},"
        txt = txt[:-1] + "\n"
        F.write(txt)

        # print(loss_dict['loss_classifier'].item())
        # print(loss_dict['loss_box_reg'].item())
        # print(loss_dict['loss_mask'].item())
        # print(loss_dict['loss_objectness'].item())
        # print(loss_dict['loss_rpn_box_reg'].item())

    F.close()
        
    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device, model_name, mask_predict = False):
    N_file = file_count('./TestMask/Images/')
    L = first_num('./TestMask/Images/')
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    F = open("plot.txt","w")
    txt = ""
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        model_time = time.time() - model_time
        
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # ----------------------- print mask -----------------------

        if mask_predict == True:
            idx = [idx for idx in res.keys()][0]
            if not os.path.exists("./Mask/" + str(idx) + "/"):
                os.makedirs("./Mask/" + str(idx) + "/")
            Exist = 0

            F_img = cv2.imread("./TestMask/Images/" + str(L[idx]))
            W_F_img = np.zeros_like(F_img)
            for N, score in enumerate(res[idx]['scores']):
                if score > 0.8:
                    print(idx)
                    if res[idx]['labels'][N] == 1:
                        Exist = Exist + 1
                        original_img = cv2.imread("./TestMask/Images/" + str(L[idx]))
                        mask_img = cv2.imread("./TestMask/Masks/" + str(L[idx].split(".")[0] + "/0.png"))

                        img =  np.array((res[idx]['masks'][N] * 255)[0]).astype(np.uint8)
                        Wimg = np.zeros_like(original_img)
                        Wimg[:,:,0] = img[:,:]
                        W_F_img[:,:,0]  = W_F_img[:,:,0] | img[:,:]
                        Wimg[:,:,1] = np.array((mask_img[:,:,0] * 255)).astype(np.uint8)
                        W_F_img[:,:,1]  = W_F_img[:,:,1] | np.array((mask_img[:,:,0] * 255)).astype(np.uint8)

                        final_img = cv2.addWeighted(original_img, 0.5, Wimg, 0.5, 20)
                        maskpath = "./Mask/" + str(idx) + "/" + str(N+1) + "_" + "1" + ".png"
                        vispath = "./Mask/" + str(idx) + "/vis" + str(N+1) + "_" + "1" + ".png"
                        x1,y1,x2,y2 = np.array(res[idx]['boxes'][N])
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        text = f"benign:{round(float(score), 2)}"
                        cv2.putText(final_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(W_F_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
                        cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.rectangle(W_F_img, (x1, y1), (x2, y2), (255, 0, 0), 2)


                        cv2.imwrite(vispath,final_img)
                        cv2.imwrite(maskpath,img)
                        txt = txt + f"{maskpath},{score},{np.array(res[idx]['boxes'][N])}\n"

                    elif res[idx]['labels'][N] == 2:
                        Exist = Exist + 1
                        original_img = cv2.imread("./TestMask/Images/" + str(L[idx]))
                        mask_img = cv2.imread("./TestMask/Masks/" + str(L[idx].split(".")[0] + "/0.png"))
                        img =  np.array((res[idx]['masks'][N] * 255)[0]).astype(np.uint8)
                        Wimg = np.zeros_like(original_img)
                        Wimg[:,:,2] = img[:,:]
                        W_F_img[:,:,2]  = W_F_img[:,:,2] | img[:,:]
                        Wimg[:,:,1] = np.array((mask_img[:,:,0] * 255)).astype(np.uint8)
                        W_F_img[:,:,1]  = W_F_img[:,:,1] | np.array((mask_img[:,:,0] * 255)).astype(np.uint8)

                        final_img = cv2.addWeighted(original_img, 0.5, Wimg, 0.5, 20)
                        maskpath = "./Mask/" + str(idx) + "/" + str(N+1) + "_" + "2" + ".png"
                        vispath = "./Mask/" + str(idx) + "/vis" + str(N+1) + "_" + "2" + ".png"
                        x1,y1,x2,y2 = np.array(res[idx]['boxes'][N])
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        text = f"malignant:{round(float(score), 2)}"
                        cv2.putText(final_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.putText(W_F_img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.rectangle(final_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.rectangle(W_F_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.imwrite(vispath,final_img)
                        cv2.imwrite(maskpath,img)
                        txt = txt + f"{maskpath},{score},{np.array(res[idx]['boxes'][N])}\n"
                        
                    else:
                        print("error")
            print(idx, L[idx])
            final_vis = "./Mask/" + str(idx) + "/vis.png"
            final_visimg = cv2.addWeighted(F_img, 0.5, W_F_img, 0.5, 20)
            cv2.imwrite(final_vis,final_visimg)
                    
            if Exist == 0:
                print(idx, "not any mask file!!!")
        

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    F.write(txt)
    F.close()
    

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize(model_name)
    torch.set_num_threads(n_threads)
    return coco_evaluator

def file_count(dir):
    initial_count = 0
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            initial_count += 1
    return initial_count

def first_num(dir):
    L_paths = sorted(os.listdir(dir))
    L = [path for path in L_paths]
    return L