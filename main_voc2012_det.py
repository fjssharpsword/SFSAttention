# encoding: utf-8
"""
Training implementation for VOC2012 dataset  
Author: Jason.Fang
Update time: 27/09/2021
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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
#define by myself
from utils.common import compute_iou, count_bytes
from utils.voc2coco import voc2coco_target
from nets.resnet_det import resnet18
from nets.densenet import densenet121

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
MAX_EPOCHS = 100
BATCH_SIZE = 1#8
root = '/data/fjsdata/VOC2012/'
VOC_CLASSES = ['background','aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
CKPT_PATH = '/data/pycode/SFSAttention/ckpts/voc2012_resnet.pkl'
def collate_fn(batch):
    return tuple(zip(*batch))
def Train():
    print('********************load data********************')
    trans = transforms.Compose([transforms.Resize((604,604)),transforms.ToTensor()])
    train_set = dset.VOCDetection(root=root, year='2012', image_set='train', transform=trans, download=False)
    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=1, collate_fn=collate_fn)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    resnet = resnet18(pretrained=False, num_classes=len(VOC_CLASSES)).cuda()
    backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    #backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 512 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=len(VOC_CLASSES), rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()
    
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
            for batch_idx, (images, targets) in enumerate(train_loader):
                optimizer_model.zero_grad()
                images = list(image.cuda() for image in images)
                targets = voc2coco_target(targets)
                targets = [{k:v.cuda() for k, v in t.items()} for t in targets]
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

        if (epoch+1) % 10 == 0:
            Test()
       
        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    trans = transforms.Compose([transforms.Resize((604,604)),transforms.ToTensor()])
    test_set = dset.VOCDetection(root=root, year='2012', image_set='val', transform=trans, download=False)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=1, collate_fn=collate_fn)
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    resnet = resnet18(pretrained=False, num_classes=len(VOC_CLASSES)).cuda()
    backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    #backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 512 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=len(VOC_CLASSES), rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()
    
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval() 
    print('********************load model succeed!********************')

    print('******* begin testing!*********')
    mAP = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], \
           10:[], 11:[], 12:[], 13:[], 14:[], 15:[], 16:[], 17:[], 18:[], 19:[], 20:[]} 
    time_res = []
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = list(image.cuda() for image in images)
            targets = voc2coco_target(targets)
            targets = [{k:v.cuda() for k, v in t.items()} for t in targets]
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
                    if iou_max > 0.50: #0.75, hit
                        mAP[0].append(1)
                        mAP[gt_lbl[m].item()].append(1)
                    else:
                        mAP[0].append(0)
                        mAP[gt_lbl[m].item()].append(0)

            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()

    for i in range(len(VOC_CLASSES)):
        print('The mAP of {} is {:.4f}'.format(VOC_CLASSES[i], np.mean(mAP[i])))
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    #print("\r Params of model: {}".format(count_bytes(param)) )
    #print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()