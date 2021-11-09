import os
import numpy as np
from numpy.linalg import norm
import cv2
from random import normalvariate
from math import sqrt
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import random
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import pylab
import matplotlib.patches as patches
import torch
import torch.nn as nn

def svd_compression(img, k):
    res_image = np.zeros_like(img)
    if len(img.shape) == 3: #RGB
        for i in range(img.shape[2]):
            U, Sigma, VT = np.linalg.svd(img[:,:,i])
            res_image[:, :, i] = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
    else: #gray
        U, Sigma, VT = np.linalg.svd(img)
        res_image = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
 
    return res_image

def attention_bias():

    fig, axes = plt.subplots(2,3, constrained_layout=True, figsize=(12,8))#

    #natural image
    annFile = '/data/fjsdata/mscoco/coco/annotations/instances_val2017.json'
    root = '/data/fjsdata/mscoco/coco/val2017/'
    coco=COCO(annFile)
    catIds = coco.getCatIds(catNms=['person','dog']) #'car'
    imgIds = coco.getImgIds(catIds=catIds)
    img = coco.loadImgs(imgIds[0])[0]
    annIds  = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    nat_ann = coco.loadAnns(annIds)
    nat_img = cv2.imread(root + img['coco_url'].split('/')[-1]) #cv2.imread(img['coco_url'], cv2.IMREAD_GRAYSCALE)
    x_f, y_f = 224/nat_img.shape[0], 224/nat_img.shape[1]
    nat_img = cv2.resize(nat_img,(224,224))
    axes[0,0].imshow(nat_img, aspect="auto") #cmap='gray'
    for ann in nat_ann:
        box = ann['bbox']
        x, y, w, h = box[0]*x_f, box[1]*y_f, box[2]*x_f, box[3]*y_f
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
        axes[0,0].add_patch(rect)# Add the patch to the Axes
    axes[0,0].set_title('(a)') 
    axes[0,0].axis('off')
    #intra-sample attention
    nat_att = svd_compression(nat_img, k=1)
    nat_att = np.mean(nat_att, axis=2)
    x = np.arange(0,nat_att.shape[1],1)
    y = np.arange(nat_att.shape[0]-1,0-1,-1)
    X,Y = np.meshgrid(x,y)
    axes[0,1].contourf(X,Y,nat_att,6,cmap="YlGnBu") 
    for ann in nat_ann:
        box = ann['bbox']
        x, y, w, h = box[0]*x_f, box[1]*y_f, box[2]*x_f, box[3]*y_f
        rect = patches.Rectangle((x, nat_att.shape[0]-y), w, -h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
        axes[0,1].add_patch(rect)# Add the patch to the Axes
    axes[0,1].set_title('(b)') 
    axes[0,1].axis('off')
    #inter-sample attention
    """
    batch_img = torch.FloatTensor()
    for i in range(len(imgIds)):
        img = coco.loadImgs(imgIds[i])[0]
        nat_img = cv2.imread(root + img['coco_url'].split('/')[-1])
        nat_img = cv2.resize(nat_img,(56,56))
        batch_img = torch.cat((batch_img, torch.Tensor(img).unsqueeze(0)), 0)
    """


    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/att_bias.png', dpi=300, bbox_inches='tight')
    

if __name__ == "__main__":

    attention_bias()