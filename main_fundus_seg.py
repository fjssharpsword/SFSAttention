# encoding: utf-8
"""
Training implementation for LIDC-IDRI CT dataset - Segmentation - MANet and HamNet
Author: Jason.Fang
Update time: 08/07/2022
"""
import re
import sys
import os
import cv2
import time
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from thop import profile
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import seaborn as sns
#define by myself
from utils.common import DiceLoss, count_bytes
from dsts.fundus_seg import get_train_dataloader, get_test_dataloader
from EMANet.network import EMANet

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
BATCH_SIZE = 6*2
MAX_EPOCHS = 100
CKPT_PATH = '/data/pycode/SFSAttention/ckpts/fundus_seg_emanet.pkl'
#nohup python3 main_fundus_seg.py > logs/main_fundus_seg.log 2>&1 &

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    dataloader_test = get_test_dataloader(batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    print ('==>>> total trainning batch number: {}'.format(len(dataloader_train)))
    print ('==>>> total test batch number: {}'.format(len(dataloader_test)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = EMANet(n_classes=2, n_layers=50).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of Fundus dataset: "+CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    log_writer = SummaryWriter('/data/tmpexec/tb_log') 
    loss_min = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, mask) in enumerate(dataloader_train):
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask).cuda()
                loss_tensor, _ = model(var_image, var_mask)

                optimizer_model.zero_grad()
                loss_tensor =loss_tensor.mean()
                loss_tensor.backward()
                optimizer_model.step()#update parameters
                
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        #test
        model.eval()
        dice_coe = []
        with torch.autograd.no_grad():
            for batch_idx,  (image, mask) in enumerate(dataloader_test):
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask).cuda()
                var_out = model(var_image)
                pred = F.log_softmax(var_out.cpu().data, dim=1)
                pred = pred.argmax(1)
                dice_coe.append(DiceLoss(pred, mask).item())
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        print("\r Eopch: %5d dice loss = %.6f" % (epoch + 1, np.mean(dice_coe)) )

        #save checkpoint with lowest loss 
        if loss_min > np.mean(dice_coe):
            loss_min = np.mean(dice_coe)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            #torch.save(model.state_dict(), CKPT_PATH)
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

        #print the loss
        log_writer.add_scalars('EMANet/Fundus_Seg', {'train':np.mean(train_loss), 'val':np.mean(dice_coe)}, epoch+1)
    log_writer.close()

def Test():
    print('********************load data********************')
    dataloader_test = get_test_dataloader(batch_size=8, shuffle=False, num_workers=1) #BATCH_SIZE
    print ('==>>> total test batch number: {}'.format(len(dataloader_test)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = EMANet(n_classes=2, n_layers=50).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of Fundus dataset: "+CKPT_PATH)
    model.eval()
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    time_res = []
    dice_coe = []
    with torch.autograd.no_grad():
        for batch_idx, (image, mask) in enumerate(dataloader_test):
            var_image = torch.autograd.Variable(image).cuda()
            start = time.time()
            var_out = model(var_image)
            end = time.time()
            time_res.append(end-start)
            pred = var_out.cpu().data
            #pred = torch.where(var_out.cpu().data>0.5, 1, 0)
            dice_coe.append(DiceLoss(pred, mask).item())
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    #model
    param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print("\r Params of model: {}".format(count_bytes(param)) )
    print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )
    #Compute Dice coefficient
    print("\r Dice coefficient = %.4f" % (1-np.mean(dice_coe)))

def main():
    Train()
    Test()

if __name__ == '__main__':
    main() 