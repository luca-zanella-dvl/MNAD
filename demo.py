import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax2d
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from model.utils import CustomDataset, DataLoader
from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *
from model.Reconstruction import *
from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse

from tqdm import tqdm


parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument("--gpus", nargs="+", type=str, help="gpus")
parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
parser.add_argument(
    "--test_batch_size", type=int, default=1, help="batch size for test"
)
parser.add_argument("--h", type=int, default=256, help="height of input images")
parser.add_argument("--w", type=int, default=256, help="width of input images")
parser.add_argument("--c", type=int, default=3, help="channel of input images")
parser.add_argument(
    "--method", type=str, default="pred", help="The target task for anoamly detection"
)
parser.add_argument(
    "--t_length", type=int, default=5, help="length of the frame sequences"
)
parser.add_argument(
    "--fdim", type=int, default=512, help="channel dimension of the features"
)
parser.add_argument(
    "--mdim", type=int, default=512, help="channel dimension of the memory items"
)
parser.add_argument("--msize", type=int, default=10, help="number of the memory items")
parser.add_argument(
    "--alpha", type=float, default=0.6, help="weight for the anomality score"
)
parser.add_argument(
    "--th", type=float, default=0.01, help="threshold for test updating"
)
parser.add_argument(
    "--num_workers", type=int, default=2, help="number of workers for the train loader"
)
parser.add_argument(
    "--num_workers_test",
    type=int,
    default=1,
    help="number of workers for the test loader",
)
parser.add_argument(
    "--dataset_type",
    type=str,
    default="ped2",
    help="type of dataset: ped2, avenue, shanghai, bg, mt",
)
parser.add_argument(
    "--dataset_path", type=str, default="./dataset", help="directory of data"
)
parser.add_argument("--model_dir", type=str, help="directory of model")
parser.add_argument("--m_items_dir", type=str, help="directory of model")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    gpus = ",".join(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

torch.backends.cudnn.enabled = (
    True  # make sure to use cudnn for computational performance
)

test_folder = os.path.join(args.dataset_path, args.dataset_type, "testing/frames")

# Loading dataset
if args.dataset_type == "bg" or args.dataset_type.startswith("mt"):
    test_dataset = CustomDataset(
        test_folder,
        transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
        resize_height=args.h,
        resize_width=args.w,
        time_step=args.t_length - 1,
    )
else:
    test_dataset = DataLoader(
        test_folder,
        transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
        resize_height=args.h,
        resize_width=args.w,
        time_step=args.t_length - 1,
    )

test_size = len(test_dataset)
anomaly_scores = np.zeros(test_size)

test_batch = data.DataLoader(
    test_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=args.num_workers_test,
    drop_last=False,
)

loss_func_mse = nn.MSELoss(reduction="none")

# Loading the trained model
model = torch.load(args.model_dir)

model.cuda()
m_items = torch.load(args.m_items_dir)

print("Evaluation of", args.dataset_type)

output_dir = os.path.join("exp", args.dataset_type, args.method, "demo")

psnr_list = []
feature_distance_list = []

m_items_test = m_items.clone()

model.eval()

for i, (imgs, frame_name) in enumerate(test_batch):
    imgs = Variable(imgs).cuda()
    frame_name = frame_name[0]

    if args.method == "pred":
        (
            outputs,
            feas,
            updated_feas,
            m_items_test,
            softmax_score_query,
            softmax_score_memory,
            _,
            _,
            _,
            compactness_loss,
        ) = model.forward(imgs[:, 0 : 3 * 4], m_items_test, False)
        mse_imgs = torch.mean(
            loss_func_mse((outputs[0] + 1) / 2, (imgs[0, 3 * 4 :] + 1) / 2)
        ).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs[:, 3 * 4 :])
    else:
        (
            outputs,
            feas,
            updated_feas,
            m_items_test,
            softmax_score_query,
            softmax_score_memory,
            compactness_loss,
        ) = model.forward(imgs, m_items_test, False)
        mse_imgs = torch.mean(
            loss_func_mse((outputs[0] + 1) / 2, (imgs[0] + 1) / 2)
        ).item()
        mse_feas = compactness_loss.item()

        # Calculating the threshold for updating at the test time
        point_sc = point_score(outputs, imgs)

    anomaly_scores[i] = point_sc
    # if i - (args.t_length - 1) >= 0:
    #     visualize_frame_with_text(frame_name, anomaly_scores[i - (args.t_length - 1)], output_dir)
    # else:
    #     visualize_frame_with_text(frame_name, -1, output_dir)

    if point_sc < args.th:
        query = F.normalize(feas, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        m_items_test = model.memory.update(query, m_items_test, False)

    psnr_list.append(psnr(mse_imgs))
    feature_distance_list.append(mse_feas)

# Measuring the abnormality score and the AUC
anomaly_score_total_list = []
anomaly_score_total_list += score_sum(
    anomaly_score_list(psnr_list),
    anomaly_score_list_inv(feature_distance_list),
    args.alpha,
)
# print(anomaly_score_total_list)

pbar = tqdm(
    total=len(test_batch),
    bar_format="{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]",
)
for i, (imgs, frame_name) in enumerate(test_batch):
    imgs = Variable(imgs).cuda()
    frame_name = frame_name[0]

    if i - (args.t_length - 1) >= 0:
        visualize_frame_with_text(frame_name, anomaly_score_total_list[i - (args.t_length - 1)], output_dir)
    else:
        visualize_frame_with_text(frame_name, -1, output_dir)

    pbar.update(1)

pbar.close()