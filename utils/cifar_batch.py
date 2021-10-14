import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from thop import profile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import seaborn as sns
import torchvision.datasets as dset
import matplotlib.image as mpimg

def vis_batch_performance():
    
    x_ticks = ['8', '16', '32', '64', '128'] #x_axies
    x1 = np.arange(len(x_ticks))
    barWidth = 0.25
    x2 = [x + barWidth for x in x1]
    #ResNet-top1
    resnet_top1 = [77.04, 77.85, 77.34, 76.63, 74.84] #y_axies
    resnet_sna_top1 = [78.38, 79.17, 78.44, 77.29, 74.97]
    gain_top1 = [1.34, 1.32, 1.10, 0.66, 0.13]
    #ResNet-top5
    resnet_top5 = [93.13, 93.69, 93.60, 93.50, 92.23] #y_axies
    resnet_sna_top5 = [93.67, 94.13, 93.99, 93.86, 92.58]
    gain_top5 = [0.54, 0.44, 0.39, 0.36, 0.35]

    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10,6)) #

    #plot resnet-top1
    axes[0,0].bar(x1, resnet_top1, color='g', width=barWidth, label='ResNet-18')
    for a, b in zip(x1, resnet_top1):
        axes[0,0].text(a+barWidth/2, b, b, ha='left', va='top', color='g')
    axes[0,0].bar(x1, gain_top1, color='m', width=barWidth, bottom=resnet_top1, label='ResNet-18 + SNA (Ours)')
    for a, b in zip(x1, resnet_sna_top1):
        axes[0,0].text(a+barWidth/2, b, b, ha='left', va='bottom', color='m')
    for a, b, c, d in zip(x1, gain_top1, resnet_top1, resnet_sna_top1):
        axes[0,0].text(a, (c+d)/2, '%.2f'%(b), ha='center', va='top', color='k')
    axes[0,0].fill_between(x1, resnet_top1, resnet_sna_top1, color='b', alpha=0.3)
    axes[0,0].set_xlabel('Batch size')
    axes[0,0].set_xticks(x1)
    axes[0,0].set_xticklabels(x_ticks)
    axes[0,0].set_ylabel('Top-1')
    axes[0,0].set_ylim(74.50,79.50)
    axes[0,0].grid(b=True, ls=':')
    axes[0,0].legend(loc = 'lower center') #upper center, lower left

    #plot resnet-top1
    axes[0,1].bar(x1, resnet_top5, color='g', width=barWidth, label='ResNet-18')
    for a, b in zip(x1, resnet_top5):
        axes[0,1].text(a+barWidth/2, b, b, ha='left', va='top', color='g')
    axes[0,1].bar(x1, gain_top5, color='m', width=barWidth, bottom=resnet_top5, label='ResNet-18 + SNA (Ours)')
    for a, b in zip(x1, resnet_sna_top5):
        axes[0,1].text(a+barWidth/2, b, b, ha='left', va='bottom', color='m')
    for a, b, c, d in zip(x1, gain_top5, resnet_top5, resnet_sna_top5):
        axes[0,1].text(a, (c+d)/2, '%.2f'%(b), ha='center', va='top', color='k')
    axes[0,1].fill_between(x1, resnet_top5, resnet_sna_top5, color='b', alpha=0.3)
    axes[0,1].set_xlabel('Batch size')
    axes[0,1].set_xticks(x1)
    axes[0,1].set_xticklabels(x_ticks)
    axes[0,1].set_ylabel('Top-5')
    axes[0,1].set_ylim(92, 94.50)
    axes[0,1].grid(b=True, ls=':')
    axes[0,1].legend(loc = 'lower center') #upper center, lower left

    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/cifar_batch.png', dpi=300, bbox_inches='tight')


def main():
    vis_batch_performance()

if __name__ == '__main__':
    main()