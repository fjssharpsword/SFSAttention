# encoding: utf-8
"""
Training implementation of object detection for 2D chest x-ray
Author: Jason.Fang
Update time: 29/07/2021
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
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from tensorboardX import SummaryWriter
from thop import profile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import seaborn as sns
#define by myself
from utils.common import compute_iou, count_bytes
from dsts.vincxr_det import get_box_dataloader_VIN
from nets.resnet import resnet50
from nets.densenet import densenet121

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
CLASS_NAMES = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
               'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
BATCH_SIZE = 2
MAX_EPOCHS = 20
NUM_CLASSES =  len(CLASS_NAMES)
CKPT_PATH = '/data/pycode/SFSAttention/ckpts/vincxr_det_densenet.pkl'

def Train():
    print('********************load data********************')
    data_loader_train = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print ('==>>> total trainning batch number: {}'.format(len(data_loader_train)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    #resnet = resnet50(pretrained=False, num_classes=NUM_CLASSES).cuda()
    #backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 1024 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()
    
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
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
            for batch_idx, (images, targets) in enumerate(data_loader_train):
                optimizer_model.zero_grad()
                images = list(image.cuda() for image in images)
                targets = [{k:v.squeeze(0).cuda() for k, v in t.items()} for t in targets]
                loss_dict  = model(images,targets)   # Returns losses and detections
                loss_tensor = sum(loss for loss in loss_dict.values())
                loss_tensor.backward()
                optimizer_model.step()##update parameters
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        if loss_min > np.mean(train_loss):
            loss_min = np.mean(train_loss)
            torch.save(model.state_dict(), CKPT_PATH) #Saving checkpoint
            print(' Epoch: {} model has been already save!'.format(epoch+1))
     
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    data_loader_test = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print ('==>>> total test batch number: {}'.format(len(data_loader_test)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    #resnet = resnet50(pretrained=False, num_classes=NUM_CLASSES).cuda()
    #backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 1024 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()
    
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval() 
    print('********************load model succeed!********************')

    print('******* begin testing!*********')
    mAP = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[]}
    time_res = []
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader_test):
            images = list(image.cuda() for image in images)
            targets = [{k:v.squeeze(0).cuda() for k, v in t.items()} for t in targets]
            start = time.time()
            var_output = model(images)#forward
            end = time.time()
            time_res.append(end-start)
        
            for i in range(len(targets)):
                gt_box = targets[i]['boxes'].cpu().data
                pred_box = var_output[i]['boxes'].cpu().data
                gt_lbl = targets[i]['labels'].cpu().data
                pred_lbl = var_output[i]['labels'].cpu().data
                for m in range(gt_box.shape[0]):
                    iou_max = 0.0
                    for n in range(pred_box.shape[0]):
                        if gt_lbl[m] == pred_lbl[n]:
                            iou = compute_iou(gt_box[m], pred_box[n])
                            if iou_max < iou: iou_max =  iou
                    if iou_max > 0.4: #hit
                        mAP[0].append(1)
                        mAP[gt_lbl[m].item()].append(1)
                    else:
                        mAP[0].append(0)
                        mAP[gt_lbl[m].item()].append(0)

            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    for i in range(NUM_CLASSES):
        print('The mAP of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mAP[i])))
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    #print("\r Params of model: {}".format(count_bytes(param)) )
    #print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()