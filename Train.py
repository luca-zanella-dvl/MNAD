import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from sklearn.metrics import roc_auc_score
from utils import *
import random

import argparse

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
wandb.init(project="piazza-fiera-MNAD")

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument("--gpus", nargs="+", type=str, help="gpus")
parser.add_argument("--batch_size", type=int, default=4, help="batch size for training")
parser.add_argument(
    "--test_batch_size", type=int, default=1, help="batch size for test"
)
parser.add_argument(
    "--epochs", type=int, default=60, help="number of epochs for training"
)
parser.add_argument(
    "--loss_compact",
    type=float,
    default=0.1,
    help="weight of the feature compactness loss",
)
parser.add_argument(
    "--loss_separate",
    type=float,
    default=0.1,
    help="weight of the feature separateness loss",
)
parser.add_argument("--h", type=int, default=256, help="height of input images")
parser.add_argument("--w", type=int, default=256, help="width of input images")
parser.add_argument("--c", type=int, default=3, help="channel of input images")
parser.add_argument("--lr", type=float, default=2e-4, help="initial learning rate")
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
parser.add_argument("--exp_dir", type=str, default="log", help="directory of log")

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if args.gpus is None:
    gpus = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
else:
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]

torch.backends.cudnn.enabled = (
    True  # make sure to use cudnn for computational performance
)

train_folder = os.path.join(args.dataset_path, args.dataset_type, "training/frames")
val_folder = os.path.join(args.dataset_path, args.dataset_type, "validation/frames")

# Loading dataset
train_dataset = DataLoader(
    train_folder,
    transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    resize_height=args.h,
    resize_width=args.w,
    time_step=args.t_length - 1,
)
val_dataset = DataLoader(
    val_folder,
    transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
    resize_height=args.h,
    resize_width=args.w,
    time_step=args.t_length - 1,
)

train_size = len(train_dataset)
val_size = len(val_dataset)

train_batch = data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
)
val_batch = data.DataLoader(
    val_dataset,
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=args.num_workers_test,
    drop_last=False,
)

# Model setting
assert args.method == "pred" or args.method == "recon", "Wrong task name"
if args.method == "pred":
    from model.final_future_prediction_with_memory_spatial_sumonly_weight_ranking_top1 import *

    model = convAE(args.c, args.t_length, args.msize, args.fdim, args.mdim)
else:
    from model.Reconstruction import *

    model = convAE(
        args.c, memory_size=args.msize, feature_dim=args.fdim, key_dim=args.mdim
    )

params_encoder = list(model.encoder.parameters())
params_decoder = list(model.decoder.parameters())
params = params_encoder + params_decoder

optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

model.cuda()


# Report the training process
log_dir = os.path.join("./exp", args.dataset_type, args.method, args.exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)


loss_func_mse = nn.MSELoss(reduction="none")

loss_pix = AverageMeter()
loss_comp = AverageMeter()
loss_sep = AverageMeter()

loss_pix_v = AverageMeter()
loss_comp_v = AverageMeter()
loss_sep_v = AverageMeter()

# Training

m_items = F.normalize(
    torch.rand((args.msize, args.mdim), dtype=torch.float), dim=1
).cuda()  # Initialize the memory items

for epoch in range(args.epochs):
    labels_list = []
    model.train()

    pbar = tqdm(total=len(train_batch))
    start = time.time()
    for j, (imgs, _) in enumerate(train_batch):

        imgs = Variable(imgs).cuda()

        if args.method == "pred":
            (
                outputs,
                _,
                _,
                m_items,
                softmax_score_query,
                softmax_score_memory,
                separateness_loss,
                compactness_loss,
            ) = model.forward(imgs[:, 0:12], m_items, mode="train")
            
        else:
            (
                outputs,
                _,
                _,
                m_items,
                softmax_score_query,
                softmax_score_memory,
                separateness_loss,
                compactness_loss,
            ) = model.forward(imgs, m_items, mode="train")

        optimizer.zero_grad()
        if args.method == "pred":
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs[:, 12:]))
            
        else:
            loss_pixel = torch.mean(loss_func_mse(outputs, imgs))

        
        loss = (
            loss_pixel
            + args.loss_compact * compactness_loss
            + args.loss_separate * separateness_loss
        )

        loss.backward(retain_graph=False)
        optimizer.step()
        

        loss_pix.update(loss_pixel.item(),  1)
        loss_comp.update(args.loss_compact*compactness_loss.item(),  1)
        loss_sep.update(args.loss_separate*separateness_loss.item(),  1)

        pbar.set_postfix({
                      'Epoch': '{0} {1}'.format(epoch+1, args.exp_dir),
                      'Lr': '{:.6f}'.format(optimizer.param_groups[-1]['lr']),
                      'Pre/Rec': '{:.6f}({:.6f})'.format(loss_pixel.item(), loss_pix.avg),
                      'Comp': '{:.6f}({:.6f})'.format(compactness_loss.item(), loss_comp.avg),
                      'Sep': '{:.6f}({:.6f})'.format(separateness_loss.item(), loss_sep.avg),
                    })
        pbar.update(1)


    scheduler.step()
    pbar.close()

    wandb.define_metric("epoch")
    wandb.define_metric("Loss/*", step_metric="epoch")
    wandb.log({"Loss/Separateness": loss_sep.avg , "epoch": epoch + 1})
    wandb.log({"Loss/Compactness": loss_comp.avg, "epoch": epoch + 1})

    writer.add_scalar("Loss/Compactness", loss_comp.avg, epoch + 1)
    writer.add_scalar("Loss/Separateness", loss_sep.avg, epoch + 1)

    if args.method == "pred":
        writer.add_scalar("Loss/Prediction", loss_pix.avg, epoch + 1)
        wandb.log({"Loss/Prediction_V": loss_pix_v.avg, "epoch": epoch +1})
        
    else:
        writer.add_scalar("Loss/Reconstruction", loss_pix.avg, epoch + 1)
        wandb.log({"Loss/Reconstruction_V": loss_pix_v.avg, "epoch": epoch +1})
        
    
    
    # Validation 
    model.eval()
    with torch.no_grad():
      for k,(imgs,_) in enumerate(val_batch):
        imgs = Variable(imgs).cuda()

        if args.method == "pred":
            (
                outputs,
                _,
                separateness_loss_v,
                compactness_loss_v,
            ) = model.forward(imgs[:, 0:12], m_items, "val")

        else:
            (
                outputs,
                _,
                separateness_loss_v,
                compactness_loss_v,
            ) = model.forward(imgs, m_items, "val")

        
        if args.method == "pred":
            loss_pixel_v = torch.mean(loss_func_mse(outputs, imgs[:, 12:]))
        else:
            loss_pixel_v = torch.mean(loss_func_mse(outputs, imgs))
        
        loss_pix_v.update(loss_pixel_v.item(),  1)
        loss_comp_v.update(args.loss_compact*compactness_loss_v.item(),  1)
        loss_sep_v.update(args.loss_separate*separateness_loss_v.item(),  1)
    
    
    wandb.log({"Loss/Separateness_V": loss_sep_v.avg, "epoch": epoch +1})
    wandb.log({"Loss/Compactness_V": loss_comp_v.avg, "epoch": epoch +1})

    writer.add_scalar("Loss/Compactness_v", loss_comp_v.avg, epoch + 1)
    writer.add_scalar("Loss/Separateness_v", loss_sep_v.avg, epoch + 1)

    if args.method == "pred":
        writer.add_scalar("Loss/Prediction_v", loss_pix_v.avg, epoch + 1)
        wandb.log({"Loss/Prediction_V": loss_pix_v.avg, "epoch": epoch +1})
        
        
    else:
        writer.add_scalar("Loss/Reconstruction_v", loss_pix_v.avg, epoch + 1)
        wandb.log({"Loss/Reconstruction_V": loss_pix_v.avg, "epoch": epoch +1})

   
    
    loss_pix.reset()
    loss_sep.reset()
    loss_comp.reset()
    loss_pix_v.reset()
    loss_sep_v.reset()
    loss_comp_v.reset()

    # Save the model
    if epoch%10==0:
      
      if len(args.gpus[0])>1:
        model_save = model.module
      else:
        model_save = model
        
      torch.save(model_save, os.path.join(log_dir, 'model_'+str(epoch)+'.pth'))

print("Training is finished")
# Save the model and the memory items
if len(args.gpus[0])>1:
    model_save = model.module
else:
    model_save = model
torch.save(model_save, os.path.join(log_dir, 'model_'+str(epoch)+'.pth'))
torch.save(m_items, os.path.join(log_dir, 'keys_'+str(epoch)+'.pt'))



wandb.finish()
writer.flush()
writer.close()

