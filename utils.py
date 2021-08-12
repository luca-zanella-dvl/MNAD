import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score
from pathlib import Path

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2) # ||ˆI^{ij}_t − I^{ij}_t ||_2
    normal = (1-torch.exp(-error)) # 1 - exp(-||ˆI^{ij}_t − I^{ij}_t ||_2)
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item() # E_t
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def anomaly_score_list_inv(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score_inv(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
        
    return anomaly_score_list

def AUC(anomal_scores, labels):
    frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    return frame_auc

def score_sum(list1, list2, alpha):
    list_result = []
    for i in range(len(list1)):
        list_result.append((alpha*list1[i]+(1-alpha)*list2[i]))
        
    return list_result

def visualize_frame_with_text(f_name, score, output_dir="output"):
    filename, file_extension = os.path.splitext(f_name)
    f_idx = int(filename.split("/")[-1])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, f"{f_idx:04}{file_extension}")

    frame = cv2.imread(f_name)

    if score >= 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        threshold = 0.5
        bgr_red = (0, 0, 255)
        bgr_green = (0, 255, 0)
        # font_color = bgr_red if score > threshold else bgr_green
        line_type = 2

        ano_description = "Anomaly" if score < threshold else "Normal"
        font_color = bgr_red if ano_description == "Anomaly" else bgr_green
        
        height, _, _ = frame.shape
        margin = 10
        bottom_left_corner_of_text = (margin, height - margin)

        cv2.putText(
            frame,
            f"{score:.2f}",
            bottom_left_corner_of_text,
            font,
            font_scale,
            font_color,
            line_type,
        )

        text_width, text_height = cv2.getTextSize("{:.2f}".format(score), font, font_scale, line_type)[0]
        ano_corner = (margin * 2 + text_width, height - margin)

        cv2.putText(
            frame,
            ano_description,
            ano_corner,
            font,
            font_scale,
            font_color,
            line_type,
        )

    # Display the image
    # cv2.imshow("frame", frame)
    cv2.imwrite(output_path, frame)
    cv2.waitKey(0)