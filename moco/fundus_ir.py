#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import vits
from dataset.Fundus import get_train_dataset_fundus, get_test_dataset_fundus
import moco.builder
import moco.loader
import moco.optimizer
from functools import partial

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
#parser.add_argument('data', metavar='DIR',  help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=1, type=int,
                    help='GPU id to use.')
# additional configs:
parser.add_argument('--pretrained', default='/data/pycode/SFSAttention/moco/logs/checkpoint_0099.pth.tar', type=str,
                    help='path to moco pretrained checkpoint')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        # Simply call main_worker function
        main_worker(args.gpu, args)
    else:
        return


def main_worker(gpu, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            256, 4096, 1.0)
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
            256, 4096, 1.0)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    cudnn.benchmark = True

    # Data loading code
   
    train_dataset = get_train_dataset_fundus()
    test_dataset = get_test_dataset_fundus()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    

    ir_validate(train_loader, val_loader, model, args)

def ir_validate(train_loader, val_loader, model, args):
    # switch to evaluate mode
    model.eval()
    device = args.gpu
    #for retrieval evaluation
    print('********************Build feature for trainset!********************')
    tr_label = torch.FloatTensor()
    tr_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(train_loader):
            tr_label = torch.cat((tr_label, label), 0)
            var_feat = model.predictor(model.base_encoder(image.to(device)))
            tr_feat = torch.cat((tr_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('********************Extract feature for testset!********************')
    te_label = torch.FloatTensor()
    te_feat = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(val_loader):
            te_label = torch.cat((te_label, label), 0)
            var_feat = model.predictor(model.base_encoder(image.to(device)))
            te_feat = torch.cat((te_feat, var_feat.cpu().data.view(var_feat.shape[0],-1)), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.numpy(), tr_feat.numpy())
    te_label = te_label.numpy()
    tr_label = tr_label.numpy()

    for topk in [5, 10, 20]:
        mHRs = {0: [], 1: [], 2: [], 3: [], 4: []}  # Hit Ratio
        mHRs_avg = []
        mAPs = {0: [], 1: [], 2: [], 3: [], 4: []}  # mean average precision
        mAPs_avg = []
        # NDCG: lack of ground truth ranking labels
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i, :].tolist()), key=lambda x: x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            te_idx = np.where(te_label[i, :] == 1)[0][0]
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = np.where(tr_label[j, :] == 1)[0][0]
                if tr_idx == te_idx:  # hit
                    num_pos = num_pos + 1
                    mAP.append(num_pos / rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs[te_idx].append(np.mean(mAP))
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs[te_idx].append(0)
                mAPs_avg.append(0)
            mHRs[te_idx].append(num_pos / rank_pos)
            mHRs_avg.append(num_pos / rank_pos)
            sys.stdout.write('\r test set process: = {}'.format(i + 1))
            sys.stdout.flush()

        CLASS_NAMES = ['Normal', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR']
        # Hit ratio
        for i in range(len(CLASS_NAMES)):
            print('Fundus mHR of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mHRs[i])))
        print("Fundus Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        # average precision
        for i in range(len(CLASS_NAMES)):
            print('Fundus mAP of {} is {:.4f}'.format(CLASS_NAMES[i], np.mean(mAPs[i])))
        print("Fundus Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))

if __name__ == '__main__':
    main()

    #nohup python fundus_ir.py > logs/train.log 2>&1 &
