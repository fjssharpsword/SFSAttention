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

    #DenseNet-top1
    densenet_top1 = [76.62, 79.31, 77.97, 77.64, 76.40] #y_axies
    densenet_sna_top1 = [78.04, 80.81, 78.74, 78.36, 76.86]
    densenet_gain_top1 = [1.6, 1.5, 0.77, 0.72, 0.46]
    #DenseNet-top5
    densenet_top5 = [93.69, 94.83, 94.57, 93.76, 93.06] #y_axies
    densenet_sna_top5 = [94.08, 95.20, 94.88, 94.02, 93.21]
    densenet_gain_top5 = [0.39, 0.37, 0.31, 0.26, 0.15]


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

    #plot resnet-top5
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

    #plot densenet-top1
    axes[1,0].bar(x1, densenet_top1, color='g', width=barWidth, label='DenseNet-121')
    for a, b in zip(x1, densenet_top1):
        axes[1,0].text(a+barWidth/2, b, b, ha='left', va='top', color='g')
    axes[1,0].bar(x1, densenet_gain_top1, color='m', width=barWidth, bottom=densenet_top1, label='DenseNet-121 + SNA (Ours)')
    for a, b in zip(x1, densenet_sna_top1):
        axes[1,0].text(a+barWidth/2, b, b, ha='left', va='bottom', color='m')
    for a, b, c, d in zip(x1, densenet_gain_top1, densenet_top1, densenet_sna_top1):
        axes[1,0].text(a, (c+d)/2, '%.2f'%(b), ha='center', va='top', color='k')
    axes[1,0].fill_between(x1, densenet_top1, densenet_sna_top1, color='b', alpha=0.3)
    axes[1,0].set_xlabel('Batch size')
    axes[1,0].set_xticks(x1)
    axes[1,0].set_xticklabels(x_ticks)
    axes[1,0].set_ylabel('Top-1')
    axes[1,0].set_ylim(76.0,81.0)
    axes[1,0].grid(b=True, ls=':')
    axes[1,0].legend(loc = 'lower center') #upper center, lower left

    #plot densenet-top5
    axes[1,1].bar(x1, densenet_top5, color='g', width=barWidth, label='DenseNet-121')
    for a, b in zip(x1, densenet_top5):
        axes[1,1].text(a+barWidth/2, b, b, ha='left', va='top', color='g')
    axes[1,1].bar(x1, densenet_gain_top5, color='m', width=barWidth, bottom=densenet_top5, label='DenseNet-121 + SNA (Ours)')
    for a, b in zip(x1, densenet_sna_top5):
        axes[1,1].text(a+barWidth/2, b, b, ha='left', va='bottom', color='m')
    for a, b, c, d in zip(x1, densenet_gain_top5, densenet_top5, densenet_sna_top5):
        axes[1,1].text(a, (c+d)/2, '%.2f'%(b), ha='center', va='top', color='k')
    axes[1,1].fill_between(x1, densenet_top5, densenet_sna_top5, color='b', alpha=0.3)
    axes[1,1].set_xlabel('Batch size')
    axes[1,1].set_xticks(x1)
    axes[1,1].set_xticklabels(x_ticks)
    axes[1,1].set_ylabel('Top-5')
    axes[1,1].set_ylim(92.50, 95.50)
    axes[1,1].grid(b=True, ls=':')
    axes[1,1].legend(loc = 'lower center') #upper center, lower left

    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/cifar_batch.png', dpi=300, bbox_inches='tight')

def vis_batch_performance2():
    
    x_ticks = ['8', '16', '32', '64', '128'] #x_axies
    x_axies = np.arange(len(x_ticks))

    #ResNet-top1
    resnet_top1 = [77.04, 77.85, 77.34, 76.63, 74.84] #y_axies
    resnet_se_top1 = [77.38, 78.04, 77.71, 76.81, 75.01]
    resnet_eca_top1 = [77.71, 78.23, 78.12, 77.09, 75.42]
    resnet_aa_top1 = [77.94, 78.37, 78.26, 77.12, 75.53]
    resnet_sna_top1 = [78.38, 79.17, 78.44, 77.29, 75.87]

    #ResNet-top5
    resnet_top5 = [93.13, 93.69, 93.60, 93.50, 92.23] #y_axies
    resnet_se_top5 = [93.11, 93.65, 93.64, 93.42, 92.20]
    resnet_eca_top5 = [93.19, 93.91, 93.70, 93.43, 92.35]
    resnet_aa_top5 = [93.32, 93.76, 93.54, 93.39, 92.31]
    resnet_sna_top5 = [93.67, 94.13, 93.99, 93.86, 92.58]


    #DenseNet-top1
    densenet_top1 = [76.62, 79.31, 77.97, 77.64, 76.40] #y_axies
    densenet_se_top1 = [77.02, 79.84, 78.43, 77.91, 76.97]
    densenet_eca_top1 = [77.23, 79.93, 78.61, 78.05, 77.12]
    densenet_aa_top1 = [77.51, 80.06, 78.73, 78.38, 77.40]
    densenet_sna_top1 = [78.04, 80.81, 78.74, 78.36, 77.86]

    #DenseNet-top5
    densenet_top5 = [93.69, 94.83, 94.57, 93.76, 93.06] #y_axies
    densenet_se_top5 = [93.57, 94.79, 94.55, 93.79, 93.11]
    densenet_eca_top5 = [93.81, 94.91, 94.63, 93.81, 93.24]
    densenet_aa_top5 = [93.87, 95.03, 94.71, 93.90, 93.35]
    densenet_sna_top5 = [94.08, 95.20, 94.88, 94.02, 93.51]


    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10,6)) #

    #plot resnet-top1
    axes[0,0].plot(x_axies, resnet_top1,'bo-',label='ResNet-18')
    axes[0,0].plot(x_axies, resnet_se_top1,'go-',label='ResNet-18 + SE')
    axes[0,0].plot(x_axies, resnet_eca_top1,'yo-',label='ResNet-18 + ECA')
    axes[0,0].plot(x_axies, resnet_aa_top1,'co-',label='ResNet-18 + AA')
    axes[0,0].plot(x_axies, resnet_sna_top1,'ro-',label='ResNet-18 + SNA (Ours)')
    for a, b in zip(x_axies, resnet_sna_top1):
        axes[0,0].text(a, b, b, ha='center', va='bottom')
    axes[0,0].set_xlabel('Batch size')
    axes[0,0].set_xticks(x_axies)
    axes[0,0].set_xticklabels(x_ticks)
    axes[0,0].set_ylabel('Top-1')
    axes[0,0].set_ylim(74.50,79.50)
    axes[0,0].grid(b=True, ls=':')
    axes[0,0].legend(loc = 'lower center') #upper center, lower left

    #plot resnet-top5
    axes[0,1].plot(x_axies, resnet_top5,'bo-',label='ResNet-18')
    axes[0,1].plot(x_axies, resnet_se_top5,'go-',label='ResNet-18 + SE')
    axes[0,1].plot(x_axies, resnet_eca_top5,'yo-',label='ResNet-18 + ECA')
    axes[0,1].plot(x_axies, resnet_aa_top5,'co-',label='ResNet-18 + AA')
    axes[0,1].plot(x_axies, resnet_sna_top5,'ro-',label='ResNet-18 + SNA (Ours)')
    for a, b in zip(x_axies, resnet_sna_top5):
        axes[0,1].text(a, b, b, ha='center', va='bottom')
    axes[0,1].set_xlabel('Batch size')
    axes[0,1].set_xticks(x_axies)
    axes[0,1].set_xticklabels(x_ticks)
    axes[0,1].set_ylabel('Top-5')
    axes[0,1].set_ylim(92, 94.50)
    axes[0,1].grid(b=True, ls=':')
    axes[0,1].legend(loc = 'lower center') #upper center, lower left

    #plot densenet-top1
    axes[1,0].plot(x_axies, densenet_top1,'bo-',label='DenseNet-18')
    axes[1,0].plot(x_axies, densenet_se_top1,'go-',label='DenseNet-18 + SE')
    axes[1,0].plot(x_axies, densenet_eca_top1,'yo-',label='DenseNet-18 + ECA')
    axes[1,0].plot(x_axies, densenet_aa_top1,'co-',label='DenseNet-18 + AA')
    axes[1,0].plot(x_axies, densenet_sna_top1,'ro-',label='DenseNet-18 + SNA (Ours)')
    for a, b in zip(x_axies, densenet_sna_top1):
        axes[1,0].text(a, b, b, ha='center', va='bottom')
    axes[1,0].set_xlabel('Batch size')
    axes[1,0].set_xticks(x_axies)
    axes[1,0].set_xticklabels(x_ticks)
    axes[1,0].set_ylabel('Top-1')
    axes[1,0].set_ylim(76.0,81.0)
    axes[1,0].grid(b=True, ls=':')
    axes[1,0].legend(loc = 'lower center') #upper center, lower left

    #plot densenet-top5
    axes[1,1].plot(x_axies, densenet_top5,'bo-',label='DenseNet-18')
    axes[1,1].plot(x_axies, densenet_se_top5,'go-',label='DenseNet-18 + SE')
    axes[1,1].plot(x_axies, densenet_eca_top5,'yo-',label='DenseNet-18 + ECA')
    axes[1,1].plot(x_axies, densenet_aa_top5,'co-',label='DenseNet-18 + AA')
    axes[1,1].plot(x_axies, densenet_sna_top5,'ro-',label='DenseNet-18 + SNA (Ours)')
    for a, b in zip(x_axies, densenet_sna_top5):
        axes[1,1].text(a, b, b, ha='center', va='bottom')
    axes[1,1].set_xlabel('Batch size')
    axes[1,1].set_xticks(x_axies)
    axes[1,1].set_xticklabels(x_ticks)
    axes[1,1].set_ylabel('Top-5')
    axes[1,1].set_ylim(92.50, 95.50)
    axes[1,1].grid(b=True, ls=':')
    axes[1,1].legend(loc = 'lower center') #upper center, lower left

    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/cifar_batch.png', dpi=300, bbox_inches='tight')

def main():
    #vis_batch_performance()
    vis_batch_performance2()

if __name__ == '__main__':
    main()