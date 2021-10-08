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

def power_iteration(W, eps=1e-10):
        """
        power iteration for max_singular_value
        """
        v = torch.FloatTensor(W.size(1), 1).normal_(0, 1)
        W_s = torch.matmul(W.T, W)
        while True:
            v_t = v
            v = torch.matmul(W_s, v_t)
            v = v/torch.norm(v)
            if abs(torch.dot(v.squeeze(), v_t.squeeze())) > 1 - eps: #converged
                break

        u = torch.matmul(W, v)
        s = torch.norm(u)
        u = u/s
        #return left vector, sigma, right vector
        return u, s, v

def svd_compression(img, k):
    res_image = np.zeros_like(img)
    if img.shape == 3: #RGB
        for i in range(img.shape[2]):
            U, Sigma, VT = np.linalg.svd(img[:,:,i])
            res_image[:, :, i] = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
    else: #gray
        U, Sigma, VT = np.linalg.svd(img)
        res_image = U[:,:k].dot(np.diag(Sigma[:k])).dot(VT[:k,:])
 
    return res_image

def plot_svd_compression():
    NATURAL_IMG_PATH = '/data/fjsdata/ImageNet/ILSVRC2012_data/val/n02129165/ILSVRC2012_val_00003788.JPEG'
    MEDICAL_IMG_PATH = '/data/fjsdata/Vin-CXR/train_val_jpg/afb6230703512afc370f236e8fe98806.jpeg'
    natural_img = cv2.imread(NATURAL_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    medical_img = cv2.imread(MEDICAL_IMG_PATH, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2,3, constrained_layout=True)#figsize=(6,9)

    #natural image
    axes[0,0].imshow(natural_img, aspect="auto",cmap='gray')
    axes[0,0].axis('off')
    axes[0,0].set_title('(a)')#Natural Image (280, 415)
    """
    #explained variance
    _, Sigma, _ = np.linalg.svd(natural_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    var_sigma = var_sigma[np.nonzero(var_sigma)]
    sns.barplot(x=list(range(1,len(var_sigma)+1)), y=var_sigma, color="limegreen", ax =axes[0,1] )
    axes[0,1].set_ylabel('Explained variance (%)')
    axes[0,1].set_xlabel('Singular value')
    axes[0,1].set_title('Singular degree')
    for ind, label in enumerate(axes[0,1].xaxis.get_ticklabels()):
        if ind == 0: label.set_visible(True)
        elif ind == len(var_sigma)-1: label.set_visible(True)
        elif (ind+1) % 5 == 0:   # every 5th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    """
    #k=1
    img_com = svd_compression(natural_img, k=1)
    axes[0,1].imshow(img_com, aspect="auto",cmap='gray')
    axes[0,1].axis('off')
    axes[0,1].set_title('(b)')#'Spectral norm'
    #non-zero singular values
    _, Sigma, _ = np.linalg.svd(natural_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    var_sigma = var_sigma[np.nonzero(var_sigma)]
    img_com = svd_compression(natural_img, k=len(var_sigma))
    axes[0,2].imshow(img_com, aspect="auto",cmap='gray')
    axes[0,2].axis('off')
    axes[0,2].set_title('(c)') #'Singular values'

    #medical image
    axes[1,0].imshow(medical_img, aspect="auto",cmap='gray')
    axes[1,0].axis('off')
    #axes[1,0].set_title('Medical Image (3072, 2540)')
    """
    #explained variance
    _, Sigma, _ = np.linalg.svd(medical_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    var_sigma = var_sigma[np.nonzero(var_sigma)]
    sns.barplot(x=list(range(1,len(var_sigma)+1)), y=var_sigma, color="limegreen", ax =axes[1,1] )
    axes[1,1].set_ylabel('Explained variance (%)')
    axes[1,1].set_xlabel('Singular value')
    axes[1,1].set_title('Singular degree')
    for ind, label in enumerate(axes[1,1].xaxis.get_ticklabels()):
        if ind == 0: label.set_visible(True)
        elif ind == len(var_sigma)-1: label.set_visible(True)
        elif (ind+1) % 2 == 0:   # every 2th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    """
    #k=1
    img_com = svd_compression(medical_img, k=1)
    axes[1,1].imshow(img_com, aspect="auto",cmap='gray')
    axes[1,1].axis('off')
    #axes[1,1].set_title('Spectral norm')
    #k=1
    _, Sigma, _ = np.linalg.svd(medical_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    var_sigma = var_sigma[np.nonzero(var_sigma)]
    img_com = svd_compression(medical_img, k=len(var_sigma))
    axes[1,2].imshow(img_com, aspect="auto",cmap='gray')
    axes[1,2].axis('off')
    #axes[1,2].set_title('Non-zero SVs')
    """
    #k=1
    u, s, v = power_iteration(torch.FloatTensor(img))
    img_com = s*torch.matmul(u, v.T)
    axes[2].imshow(img_com.numpy(), aspect="auto",cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Rank 1')
    
    #overlay
    img_com = img_com.astype(np.uint8)
    img_com = cv2.applyColorMap(img_com, cv2.COLORMAP_JET)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.7, img_com, 0.3, 0)
    axes[2].imshow(overlay, aspect="auto",cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Overlay Image')
    """
    
    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/img_com.png', dpi=300, bbox_inches='tight')

def plot_svd_batch():
    
    NATURAL_IMG_PATH = '/data/fjsdata/ImageNet/ILSVRC2012_data/' #natural image
    MEDICAL_IMG_PATH = '/data/fjsdata/Vin-CXR/train_val_jpg/'#medical image
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(56),
        transforms.ToTensor()
    ])
    #calculate singular degree
    nat_sd, med_sd = [], []
    for bs in [8, 16, 32, 64]:
        #natural image
        nat_loader = torch.utils.data.DataLoader(
                        dset.ImageFolder(NATURAL_IMG_PATH+'val/', transform_test),
                        batch_size=bs, shuffle=False, num_workers=0)
        for batch_idx, (img, lbl) in enumerate(nat_loader):
            img = torch.mean(img, dim=1, keepdim=True).squeeze()
            img = img.view(img.size(0), -1)
            _, Sigma, _ = np.linalg.svd(img.numpy())
            var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
            var_sigma = var_sigma[np.nonzero(var_sigma)]
            sd = "{:.2}".format(bs/len(var_sigma)) #condition number
            nat_sd.append(sd)
            break
        #medical image
        batch_img = torch.FloatTensor()
        for _, _, fs in os.walk(MEDICAL_IMG_PATH):
            for f in fs:
                if batch_img.size(0) == bs: break
                img = os.path.join(MEDICAL_IMG_PATH, f)
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(56,56))
                batch_img = torch.cat((batch_img, torch.Tensor(img).unsqueeze(0)), 0)
        batch_img = batch_img.view(batch_img.size(0), -1)
        _, Sigma, _ = np.linalg.svd(batch_img.numpy())
        var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
        var_sigma = var_sigma[np.nonzero(var_sigma)]
        sd = "{:.2}".format(bs/len(var_sigma))
        med_sd.append(sd)
        
    #plot 
    fig, axes = plt.subplots(1,2, constrained_layout=True)
    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/batch_svd.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    """
    W = torch.rand(100,200)
    u, s, v = power_iteration(W)
    print(s)
    U, S, V = torch.svd(W)
    print(S.max())
    """
    #plot_svd_compression()
    plot_svd_batch()