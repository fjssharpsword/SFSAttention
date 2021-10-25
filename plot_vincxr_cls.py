# encoding: utf-8
"""
Training implementation for VIN-CXR dataset  
Author: Jason.Fang
Update time: 16/08/2021
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
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.patches as patches
#define by myself
from utils.common import count_bytes, compute_AUCs
from nets.resnet import resnet18
from nets.densenet import densenet121
from dsts.vincxr_cls import get_box_dataloader_VIN
from dsts.vincxr_det import get_box_dataloader_VIN as get_box_dataloader_VIN_BOX
#config
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
CLASS_NAMES = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'Interstitial lung disease', \
                'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
CKPT_PATH = '/data/pycode/SFSAttention/ckpts/vincxr_cls_resnet_sna.pkl'

def BoxTest():
    print('********************load data********************')
    test_loader = get_box_dataloader_VIN_BOX(batch_size=5, shuffle=False, num_workers=0)
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet18(pretrained=False, num_classes=len(CLASS_NAMES)).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval()#turn to test mode
    print('********************load model succeed!********************')

    print('********************begin Testing!********************')
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images_tn = torch.FloatTensor()
            for image in images:
                images_tn = torch.cat((images_tn, image.unsqueeze(0)), 0)
            var_output = model(images_tn.cuda())#forward

            np.save('/data/pycode/SFSAttention/logs/cxr_cls_resnet/resnet_sna_img.npy', images_tn.numpy())
            targets = [{k:v.squeeze(0) for k, v in t.items()} for t in targets]
            box_np = [ tgt['boxes'].numpy() for tgt in targets]
            lbl_np = [ tgt['labels'].numpy() for tgt in targets]
            np.save('/data/pycode/SFSAttention/logs/cxr_cls_resnet/resnet_sna_box.npy', np.array(box_np))
            np.save('/data/pycode/SFSAttention/logs/cxr_cls_resnet/resnet_sna_lbl.npy', np.array(lbl_np))
            break

def Heatmap():
    root = '/data/pycode/SFSAttention/logs/cxr_cls_resnet/'
    img = np.load(root + 'resnet_sna_img.npy')
    box = np.load(root + 'resnet_sna_box.npy', allow_pickle=True)
    lbl = np.load(root + 'resnet_sna_lbl.npy', allow_pickle=True)
    fea_before = np.load(root + 'resnet_sna_fea_before.npy')
    fea_after = np.load(root + 'resnet_sna_fea_after.npy')

    fig, axes = plt.subplots(3,5, constrained_layout=True, figsize=(20,9))

    #first row- origi image
    for i in range(5):
        img_ori = np.transpose(img[i],(1,2,0))
        axes[0,i].imshow(img_ori, aspect="auto")
        for j in range(len(lbl)):
            x, y, w, h = box[i][j][0], box[i][j][1], box[i][j][2]-box[i][j][0], box[i][j][3]-box[i][j][1]
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
            axes[0,i].add_patch(rect)# Add the patch to the Axes
            lbl_val = int(lbl[i][j])
            axes[0,i].text(int(x)-20, int(y)-5, CLASS_NAMES[lbl_val])
        axes[0,i].axis('off')

    #second row - heatmap befor sna layer
    for i in range(5):
        img_ori = np.transpose(img[i],(1,2,0))
        #img_ori = np.uint8(255 * img_ori) #[0,1] ->[0,255]
        fea_map = fea_before[i]
        cam = np.mean(fea_map, 0) #h,w
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        #cam = np.uint8(255 * cam)
        cam = cv2.resize(cam, (img_ori.shape[0], img_ori.shape[1]))
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        overlay_img = cv2.addWeighted(img_ori, 0.7, cam, 0.3, 0)
        axes[1,i].imshow(overlay_img, aspect="auto")
        axes[1,i].axis('off')




    #third row -heatmap after sna layer

    #save 
    fig.savefig('/data/pycode/SFSAttention/imgs/cxr_heatmap.png', dpi=300, bbox_inches='tight')

def main():
    #Train()
    #Test()
    #BoxTest()
    Heatmap()

if __name__ == '__main__':
    main()