import os
import numpy as np
from numpy.linalg import norm
import cv2
from random import normalvariate
from math import sqrt
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

class SpatialAttention(nn.Module):#spatial attention layer
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        pool_out, _ = torch.max(self.maxpool(x), dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out, pool_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

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

    fig, axes = plt.subplots(1,3, constrained_layout=True, figsize=(12,4))#

    #natural image
    annFile = '/data/fjsdata/mscoco/coco/annotations/instances_val2017.json'
    root = '/data/fjsdata/mscoco/coco/val2017/'
    coco=COCO(annFile)
    catIds = coco.getCatIds(catNms=['dog','person']) #'person', 'car', 'dog'
    imgIds = coco.getImgIds(catIds=catIds)
    imgIds = random.sample(imgIds, 2)
    img = coco.loadImgs(imgIds[0])[0]
    annIds  = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    nat_ann = coco.loadAnns(annIds)
    nat_img = cv2.imread(root + img['coco_url'].split('/')[-1], cv2.IMREAD_GRAYSCALE) #cv2.IMREAD_GRAYSCALE
    axes[0].imshow(nat_img, aspect="auto", cmap='gray') #
    for ann in nat_ann:
        box = ann['bbox']
        x, y, w, h = box[0], box[1], box[2], box[3]
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
        axes[0].add_patch(rect)# Add the patch to the Axes
    axes[0].set_title('(a)') 
    axes[0].axis('off')
    #intra-sample attention
    nat_att = svd_compression(nat_img, k=1)
    #nat_att = np.max(nat_img, axis=2) 
    x_f, y_f = 32/nat_att.shape[1], 32/nat_att.shape[0]
    nat_att = cv2.resize(nat_att,(32,32))
    #x = np.arange(0,nat_att.shape[1],1)
    #y = np.arange(nat_att.shape[0]-1,0-1,-1)
    #X,Y = np.meshgrid(x,y)
    #axes[1].contourf(X,Y,nat_att,6,cmap="YlGnBu")
    axes[1].imshow(nat_att, cmap="YlGnBu") 
    for ann in nat_ann:
        box = ann['bbox']
        #x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x, y, w, h = box[0]*x_f, box[1]*y_f, box[2]*x_f, box[3]*y_f
        #rect = patches.Rectangle((x, nat_att.shape[0]-y), w, -h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[1].add_patch(rect)# Add the patch to the Axes
        #msk_att[x:x+w, y:y+h] = nat_att[x:x+w, y:y+h]
    axes[1].set_title('(b)') 
    axes[1].axis('off')
    #inter-sample attention
    batch_img = torch.FloatTensor()
    for i in range(len(imgIds)):#
        img = coco.loadImgs(imgIds[i])[0]
        bat_img = cv2.imread(root + img['coco_url'].split('/')[-1], cv2.IMREAD_GRAYSCALE)
        bat_img = cv2.resize(bat_img,(32,32))
        batch_img = torch.cat((batch_img, torch.Tensor(bat_img).unsqueeze(0)), 0)
    batch_img = batch_img.view(batch_img.size(0), -1)
    batch_img = svd_compression(batch_img.numpy(), k=1)
    nat_att = batch_img[0,:].reshape((32,32))
    #x = np.arange(0,nat_att.shape[1],1)
    #y = np.arange(nat_att.shape[0]-1,0-1,-1)
    #X,Y = np.meshgrid(x,y)
    #axes[2].contourf(X,Y,nat_att,6,cmap="YlGnBu") 
    axes[2].imshow(nat_att, cmap="YlGnBu") 
    for ann in nat_ann:
        box = ann['bbox']
        x, y, w, h = box[0]*x_f, box[1]*y_f, box[2]*x_f, box[3]*y_f
        #rect = patches.Rectangle((x, nat_att.shape[0]-y), w, -h, linewidth=2, edgecolor='r', facecolor='none')# Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axes[2].add_patch(rect)# Add the patch to the Axes
    axes[2].set_title('(c)') 
    axes[2].axis('off')
    
    #medical image

    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/att_bias.png', dpi=300, bbox_inches='tight')
    


if __name__ == "__main__":

    #x =  torch.rand(2, 512, 10, 10).cuda()
    #sa = SpatialAttention().cuda()
    #out = sa(x)
    #print(out.shape)
    attention_bias()




