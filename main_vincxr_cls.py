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
#define by myself
from utils.common import count_bytes, compute_AUCs
from nets.resnet import resnet18
from nets.vit import ViT
import nets.deit
from timm.models import create_model
from nets.densenet import densenet121
from dsts.vincxr_cls import get_box_dataloader_VIN
#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
max_epoches = 30
BATCH_SIZE = 32*8
CLASS_NAMES = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
               'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
CKPT_PATH = '/data/pycode/SFSAttention/ckpts/vincxr_cls_deit.pkl'
#nohup python main_vincxr_cls.py > logs/vincxr_cls_deit.log 2>&1 &
def Train():
    print('********************load data********************')
    train_loader = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    #model = resnet18(pretrained=False, num_classes=len(CLASS_NAMES))
    #model = ViT(image_size = 224, patch_size = 32, num_classes = len(CLASS_NAMES),dim = 1024,depth = 6,heads = 16,mlp_dim = 2048,dropout = 0.1,emb_dropout = 0.1).cuda()
    model = create_model('deit_tiny_distilled_patch16_224',pretrained=False,num_classes=len(CLASS_NAMES),drop_rate=0.0,drop_path_rate=0.1,drop_block_rate=None).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.BCELoss().cuda() #nn.CrossEntropyLoss().cuda()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    #log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    acc_min = 0.50
    for epoch in range(max_epoches):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , max_epoches))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train = []
        with torch.autograd.enable_grad():
            for batch_idx, (img, lbl) in enumerate(train_loader):
                #forward
                var_image = torch.autograd.Variable(img).cuda()
                var_label = torch.autograd.Variable(lbl).cuda()
                var_out = model(var_image)
                # backward and update parameters
                optimizer_model.zero_grad()
                loss_tensor = criterion.forward(var_out, var_label) 
                loss_tensor.backward()
                optimizer_model.step()
                #show 
                loss_train.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(loss_train) ))

        #test
        model.eval()
        loss_test = []
        gt = torch.FloatTensor()
        pred = torch.FloatTensor()
        with torch.autograd.no_grad():
            for batch_idx,  (img, lbl) in enumerate(test_loader):
                #forward
                var_image = torch.autograd.Variable(img).cuda()
                var_label = torch.autograd.Variable(lbl).cuda()
                var_out = model(var_image)
                loss_tensor = criterion.forward(var_out, var_label)
                loss_test.append(loss_tensor.item())
                gt = torch.cat((gt, lbl), 0)
                pred = torch.cat((pred, var_out.data.cpu()), 0)
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        acc = np.mean(compute_AUCs(gt, pred, len(CLASS_NAMES)))
        print("\r Eopch: %5d val loss = %.6f, ACC = %.6f" % (epoch + 1, np.mean(loss_test), acc))

        # save checkpoint
        if acc_min < acc:
            acc_min = acc
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
        #log_writer.add_scalars('BCELoss/VINCXR-CLS-ResNet', {'Train':np.mean(loss_train), 'Test':np.mean(loss_test)}, epoch+1)
    #log_writer.close() #shut up the tensorboard

def Test():
    print('********************load data********************')
    test_loader = get_box_dataloader_VIN(batch_size=32, shuffle=False, num_workers=8)
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    #model = resnet18(pretrained=False, num_classes=len(CLASS_NAMES)).cuda()
    #model = ViT(image_size = 224, patch_size = 32, num_classes = len(CLASS_NAMES),dim = 1024,depth = 6,heads = 16,mlp_dim = 2048,dropout = 0.1,emb_dropout = 0.1).cuda()
    model = create_model('deit_tiny_distilled_patch16_224',pretrained=False,num_classes=len(CLASS_NAMES),drop_rate=0.0,drop_path_rate=0.1,drop_block_rate=None).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval()#turn to test mode
    print('********************load model succeed!********************')

    print('********************begin Testing!********************')
    time_res = []
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx,  (img, lbl) in enumerate(test_loader):
            #forward
            var_image = torch.autograd.Variable(img).cuda()
            var_label = torch.autograd.Variable(lbl).cuda()
            start = time.time()
            var_out = model(var_image)
            end = time.time()
            time_res.append(end-start)

            gt = torch.cat((gt, lbl), 0)
            pred = torch.cat((pred, var_out.data.cpu()), 0)

            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    #print("\r Params of model: {}".format(count_bytes(param)) )
    #print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )
    AUROCs = compute_AUCs(gt, pred, len(CLASS_NAMES))
    for i in range(len(CLASS_NAMES)):
        print('The AUROC of {} is {:.4f}'.format(CLASS_NAMES[i], AUROCs[i]))
    print('The average AUROC is {:.4f}'.format(np.mean(AUROCs)))
    #save
    #np.save('/data/pycode/SFSAttention/logs/cxr_cls/densenet_sna_gt_1.npy',gt.numpy()) #np.load()
    #np.save('/data/pycode/SFSAttention/logs/cxr_cls/densenet_sna_pd_1.npy',pred.numpy())

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()