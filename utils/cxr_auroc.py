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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
#sys.path.append("..") 

def vis_auroc():
    #fpr = 1-Specificity, tpr=Sensitivity
    np.set_printoptions(suppress=True) #to float
    class_names=['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'Interstitial lung disease', \
                'Infiltration', 'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
    model_names=['DenseNet-121', 'DenseNet-121+SE', 'DenseNet-121+ECA', 'DenseNet-121+AA', 'DenseNet-121+SNA(Ours)']
    root = '/data/pycode/SFSAttention/logs/cxr_cls/'

    fig, axes = plt.subplots(3,5, constrained_layout=True, figsize=(20,9))
    color_name =['b','y','c','g','r'] #color ref: https://www.cnblogs.com/darkknightzh/p/6117528.html

    for i in range(len(class_names)):
        m = i // 5 
        n = i % 5

        gt, pd = [], []
        gt.append(np.load(root + 'densenet_gt.npy')[:,i])
        pd.append(np.load(root + 'densenet_pd.npy')[:,i])
        gt.append(np.load(root + 'densenet_se_gt.npy')[:,i])
        pd.append(np.load(root + 'densenet_se_pd.npy')[:,i])
        gt.append(np.load(root + 'densenet_eca_gt.npy')[:,i])
        pd.append(np.load(root + 'densenet_eca_pd.npy')[:,i])
        gt.append(np.load(root + 'densenet_aa_gt.npy')[:,i])
        pd.append(np.load(root + 'densenet_aa_pd.npy')[:,i])
        gt.append(np.load(root + 'densenet_sna_gt.npy')[:,i])
        pd.append(np.load(root + 'densenet_sna_pd.npy')[:,i])

        for j in range(len(model_names)):
            fpr, tpr, threshold = roc_curve(np.array(gt[j]), np.array(pd[j]))
            auc_score = auc(fpr, tpr)
            axes[m,n].plot(fpr, tpr, c = color_name[j], ls = '--', label = u'{}-{:.2f}'.format(model_names[j],auc_score*100))

        axes[m,n].plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
        axes[m,n].set_xlim((-0.01, 1.02))
        axes[m,n].set_ylim((-0.01, 1.02))
        axes[m,n].set_xticks(np.arange(0, 1.1, 0.2))
        axes[m,n].set_yticks(np.arange(0, 1.1, 0.2))
        axes[m,n].set_xlabel('1-Specificity')
        axes[m,n].set_ylabel('Sensitivity')
        axes[m,n].grid(b=True, ls=':')
        axes[m,n].legend(loc='lower right')
        axes[m,n].set_title(class_names[i])

    fig.savefig('/data/pycode/SFSAttention/imgs/CXR_ROCCurve.png', dpi=300, bbox_inches='tight')

def main():
    vis_auroc()

if __name__ == '__main__':
    main()