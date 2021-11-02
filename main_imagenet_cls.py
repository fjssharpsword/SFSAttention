# encoding: utf-8
"""
Training implementation for ImageNet-1k dataset  
Author: Jason.Fang
Update time: 18/09/2021
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
from torchstat import stat
from tensorboardX import SummaryWriter
import seaborn as sns
#define by myself
from utils.common import count_bytes
from nets.resnet import resnet50
from nets.densenet import densenet121
#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
max_epoches = 150 #100
batch_size = 256 #128
CKPT_PATH = '/data/pycode/SFSAttention/ckpts/imagenet1k_resnet_sna.pkl' 
#nohup python main_imagenet_cls.py > logs/imagenet1k_resnet_sna.log 2>&1 &
DATA_PATH = '/data/fjsdata/ImageNet/ILSVRC2012_data/'
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Train():
    print('********************load data********************')
    # Normalize training set together with augmentation
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    train_loader = torch.utils.data.DataLoader(
                    dset.ImageFolder(DATA_PATH+'train/', transform_train),
                    batch_size=batch_size,
                    shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
                    dset.ImageFolder(DATA_PATH+'val/', transform_test),
                    batch_size=batch_size,
                    shuffle=False, num_workers=8)

    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total validation batch number: {}'.format(len(val_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet50(pretrained=False, num_classes=1000)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #optimizer_model = optim.Adam(model.parameters(), lr=lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5) 
    #lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    optimizer_model = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4) #lr=0.1
    lr_scheduler_model = lr_scheduler.MultiStepLR(optimizer_model, milestones=[30, 60, 90], gamma=0.2) #learning rate decay
    criterion = nn.CrossEntropyLoss().cuda()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    #log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    acc_min = 0.50 #float('inf')
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

        if (epoch+1) % 5 == 0:
            #test
            model.eval()
            loss_test = []
            total_cnt, correct_cnt = 0, 0
            with torch.autograd.no_grad():
                for batch_idx,  (img, lbl) in enumerate(val_loader):
                    #forward
                    var_image = torch.autograd.Variable(img).cuda()
                    var_label = torch.autograd.Variable(lbl).cuda()
                    var_out = model(var_image)
                    loss_tensor = criterion.forward(var_out, var_label)
                    loss_test.append(loss_tensor.item())
                    _, pred_label = torch.max(var_out.data, 1)
                    total_cnt += var_image.data.size()[0]
                    correct_cnt += (pred_label == var_label.data).sum()
                    sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                    sys.stdout.flush()
            acc = correct_cnt * 1.0 / total_cnt
            print("\r Eopch: %5d val loss = %.6f, ACC = %.6f" % (epoch + 1, np.mean(loss_test), acc) )

            # save checkpoint
            if acc_min < acc:
                acc_min = acc
                torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
                print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
        #log_writer.add_scalars('CrossEntropyLoss/CIFAR100-ResNet-SFConv', {'Train':np.mean(loss_train), 'Test':np.mean(loss_test)}, epoch+1)
    #log_writer.close() #shut up the tensorboard

def Test():
    print('********************load data********************')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test_loader = torch.utils.data.DataLoader(
                    dset.ImageFolder(DATA_PATH+'val/', transform_test),
                    batch_size=batch_size,
                    shuffle=False, num_workers=8)
    print ('==>>> total validation batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet50(pretrained=False, num_classes=1000).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval()#turn to test mode
    print('********************load model succeed!********************')
    
    print('********************begin Testing!********************')
    total_cnt, top1, top5 = 0, 0, 0
    time_res = []
    with torch.autograd.no_grad():
        for batch_idx,  (img, lbl) in enumerate(test_loader):
            #forward
            var_image = torch.autograd.Variable(img).cuda()
            var_label = torch.autograd.Variable(lbl).cuda()
            start = time.time()
            var_out = model(var_image)
            end = time.time()
            time_res.append(end-start)

            total_cnt += var_image.data.size()[0]
            _, pred_label = torch.max(var_out.data, 1) #top1
            top1 += (pred_label == var_label.data).sum()
            _, pred_label = torch.topk(var_out.data, 5, 1)#top5
            pred_label = pred_label.t()
            pred_label = pred_label.eq(var_label.data.view(1, -1).expand_as(pred_label))
            top5 += pred_label.float().sum()

            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    """
    param_size = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,'---', param.size())
            param_size = param_size + param.numel()
    param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print("\r Params of model: {}".format(count_bytes(param)) )
    #flops, params = profile(model, inputs=(var_image,))
    #print("FLOPs(Floating Point Operations) of model = {}".format(count_bytes(flops)) )
    #print("\r Params of model: {}".format(count_bytes(params)) )
    print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )
    print(stat(model.cpu(), (3,244,244)))
    """
    acc = top1 * 1.0 / total_cnt
    ci  = 1.96 * math.sqrt( (acc * (1 - acc)) / total_cnt) #1.96-95%
    print("\r Top-1 ACC/CI = %.4f/%.4f" % (acc, ci) )
    acc = top5 * 1.0 / total_cnt
    ci  = 1.96 * math.sqrt( (acc * (1 - acc)) / total_cnt) #1.96-95%
    print("\r Top-5 ACC/CI = %.4f/%.4f" % (acc, ci) )

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()