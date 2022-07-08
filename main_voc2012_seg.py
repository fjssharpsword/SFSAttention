# encoding: utf-8
"""
Training implementation for VOC2012 dataset  
Author: Jason.Fang
Update time: 08/07/2021
Ref: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html
     https://pytorch.org/vision/0.8/_modules/torchvision/datasets/voc.html
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import math
from thop import profile
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from tensorboardX import SummaryWriter
import seaborn as sns
#define by myself
from utils.common import DiceLoss, count_bytes
from EMANet.network import EMANet

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5"
BATCH_SIZE = 6*16
MAX_EPOCHS = 200
CKPT_PATH = '/data/pycode/SFSAttention/ckpts/voc2012_seg_emanet.pkl'
#nohup python3 main_voc2012_seg.py > logs/main_voc2012_seg.log 2>&1 &

def collate_fn(batch):
    return tuple(zip(*batch))
def Train():
    print('********************load data********************')
    root = '/data/fjsdata/VOC2012/'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    train_set = dset.VOCSegmentation(root=root, year='2012', image_set='train', transform=trans, download=False)
    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=1, collate_fn=collate_fn)

    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = EMANet(n_classes=21, n_layers=50).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of VOC2012 dataset: "+CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    loss_min = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (images, masks) in enumerate(train_loader):
                var_images = torch.FloatTensor()
                var_masks = torch.FloatTensor()
                for image in images:
                    var_images = torch.cat((var_images, image.unsqueeze(0)),0)
                for mask in masks:
                    var_masks = torch.cat((var_masks, trans(mask).unsqueeze(0)),0)
                var_masks = torch.as_tensor(var_masks, dtype=torch.long)

                loss_tensor, _ = model(var_images.cuda(), var_masks.cuda())
                optimizer_model.zero_grad()
                loss_tensor.backward()
                optimizer_model.step()#update parameters
                
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        #save checkpoint with lowest loss 
        if loss_min > np.mean(train_loss):
            loss_min = np.mean(train_loss)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            #torch.save(model.state_dict(), CKPT_PATH)
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    root = '/data/fjsdata/VOC2012/'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
    test_set = dset.VOCSegmentation(root=root, year='2012', image_set='val', transform=trans, download=False) 
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size= BATCH_SIZE,
                    shuffle=False, num_workers=1, collate_fn=collate_fn)
    print ('==>>> total testing batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = EMANet(n_classes=21, n_layers=50).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of VOC2012 dataset: "+CKPT_PATH)
    model.eval()
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    time_res = []
    dice_coe = []
    with torch.autograd.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            var_images = torch.FloatTensor()
            var_masks = torch.FloatTensor()
            for image in images:
                var_images = torch.cat((var_images, image.unsqueeze(0)),0)
            for mask in masks:
                var_masks = torch.cat((var_masks, trans(mask).unsqueeze(0)),0)

            start = time.time()
            var_out = model(var_images.cuda())
            end = time.time()
            time_res.append(end-start)
            pred = F.log_softmax(var_out.cpu().data, dim=1)
            pred = pred.argmax(1)
            dice_coe.append(DiceLoss(pred, var_masks).item())
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    #model
    param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print("\r Params of model: {}".format(count_bytes(param)) )
    #flops, _ = profile(model, inputs=(var_images,))
    #print("FLOPs(Floating Point Operations) of model = {}".format(count_bytes(flops)) )
    #print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )
    #Compute Dice coefficient
    print("\r Dice coefficient = %.4f" % (1-np.mean(dice_coe)))

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()