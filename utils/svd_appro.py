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

    fig, axes = plt.subplots(2,5, constrained_layout=True, figsize=(15,6))#

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
    #_, Sigma, _ = np.linalg.svd(natural_img)
    #var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    #var_sigma = var_sigma[np.nonzero(var_sigma)]
    #img_com = svd_compression(natural_img, k=len(var_sigma))
    img_com = svd_compression(natural_img, k=30)
    axes[0,2].imshow(img_com, aspect="auto",cmap='gray')
    axes[0,2].axis('off')
    axes[0,2].set_title('(c)') #'Singular values'
    #explained variance
    _, Sigma, _ = np.linalg.svd(natural_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    cum_sum = np.cumsum(var_sigma)[0:50]
    axes[0,3].plot(np.arange(len(cum_sum)), cum_sum,'g-')
    non_zero_point = var_sigma[np.nonzero(var_sigma)]
    idx = len(non_zero_point)
    axes[0,3].plot(np.arange(len(cum_sum))[idx], cum_sum[idx],'ro')
    axes[0,3].axhline(y=cum_sum[idx], xmin=0, xmax=len(non_zero_point)/len(cum_sum), color='b', linestyle='--')
    axes[0,3].axvline(x=np.arange(len(cum_sum))[idx], ymin=0, ymax=cum_sum[idx], color='b', linestyle='--')
    #axes[0,3].set_ylabel('Explained variance')
    #axes[0,3].set_xlabel('Dimensions')
    axes[0,3].set_title('(d)')
    axes[0,3].grid(b=True, ls=':')

    #batch_svd
    nat_sd = [0.86, 0.84, 0.81, 0.83, 0.79]
    x_axis = [1,2,3,4,5]
    axes[0,4].plot(x_axis, np.array(nat_sd),'go-')
    axes[0,4].set_xticks(x_axis)
    axes[0,4].set_xticklabels(['8', '16', '32', '64', '128'])
    axes[0,4].set_title('(e)')
    axes[0,4].grid(b=True, ls=':')

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
    #_, Sigma, _ = np.linalg.svd(medical_img)
    #var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    #var_sigma = var_sigma[np.nonzero(var_sigma)]
    #img_com = svd_compression(medical_img, k=len(var_sigma))
    img_com = svd_compression(medical_img, k=30)
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
    #explained variance
    _, Sigma, _ = np.linalg.svd(medical_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    cum_sum = np.cumsum(var_sigma)[0:50]
    axes[1,3].plot(np.arange(len(cum_sum)), cum_sum,'g-')
    non_zero_point = var_sigma[np.nonzero(var_sigma)]
    idx = len(non_zero_point)
    axes[1,3].plot(np.arange(len(cum_sum))[idx], cum_sum[idx],'ro')
    axes[1,3].axhline(y=cum_sum[idx], xmin=0, xmax=len(non_zero_point)/len(cum_sum), color='b', linestyle='--')
    axes[1,3].axvline(x=np.arange(len(cum_sum))[idx], ymin=0, ymax=cum_sum[idx], color='b', linestyle='--')
    #axes[1,3].set_ylabel('Explained variance')
    #axes[1,3].set_xlabel('Dimensions')
    #axes[1,3].set_title('(d)')
    axes[1,3].grid(b=True, ls=':')

    #batch_svd
    med_sd = [0.95, 0.93, 0.93, 0.92, 0.90]
    x_axis = [1,2,3,4,5]
    axes[1,4].plot(x_axis, np.array(med_sd),'go-')
    axes[1,4].set_xticks(x_axis)
    axes[1,4].set_xticklabels(['8', '16', '32', '64', '128'])
    axes[1,4].grid(b=True, ls=':')

    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/img_com.png', dpi=300, bbox_inches='tight')

def calculate_batch_SN():
    
    NATURAL_IMG_PATH = '/data/fjsdata/ImageNet/ILSVRC2012_data/' #natural image
    MEDICAL_IMG_PATH = '/data/fjsdata/Vin-CXR/train_val_jpg/'#medical image
    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(56),
        transforms.ToTensor()
    ])
    #calculate singular degree
    nat_sd, med_sd = [], []
    for bs in [8, 16, 32, 64, 128]:
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
            #sd = "{:.2}".format(bs/len(var_sigma)) #condition number
            sd = "{:.2}".format(var_sigma.max()) #spectral norm
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
        #sd = "{:.2}".format(bs/len(var_sigma))
        sd = "{:.2}".format(var_sigma.max()) #spectral norm
        med_sd.append(sd)
    print(nat_sd)
    print(med_sd)
    """
    #plot 
    nat_sd = [0.92, 0.86, 0.86, 0.84, 0.81] #['0.86', '0.84', '0.81', '0.83', '0.86']
    med_sd = [0.98, 0.94, 0.93, 0.93, 0.90] #['0.94', '0.94', '0.94', '0.94', '0.93']
    fig, axe = plt.subplots(1)
    x_axis = [1,2,3,4,5]
    axe.plot(x_axis, np.array(nat_sd),'go-',label='Natural images')
    axe.plot(x_axis, np.array(med_sd),'b^-',label='Medical images')
    axe.set_xticks(x_axis)
    axe.set_xticklabels(['2', '4', '8', '16', '32'])
    axe.set_ylabel('Spectral norm')
    axe.set_xlabel('Batch size')
    axe.set_title('Explained variance ratio')
    axe.grid(b=True, ls=':')
    axe.legend()
    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/batch_svd.png', dpi=300, bbox_inches='tight')
    """

def plot_svd_compression2():
    NATURAL_IMG_PATH = '/data/fjsdata/ImageNet/ILSVRC2012_data/val/n02129165/ILSVRC2012_val_00003788.JPEG'
    MEDICAL_IMG_PATH = '/data/fjsdata/Vin-CXR/train_val_jpg/afb6230703512afc370f236e8fe98806.jpeg'
    natural_img = cv2.imread(NATURAL_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    medical_img = cv2.imread(MEDICAL_IMG_PATH, cv2.IMREAD_GRAYSCALE)

    plt.figure(figsize=(15, 6))#constrained_layout=True

    ax1 = plt.subplot2grid((2, 5), (0, 0))
    ax1.imshow(natural_img, aspect="auto",cmap='gray')
    ax1.axis('off')
    ax1.set_title('(a)')#Natural Image (280, 415)
    ax2 = plt.subplot2grid((2, 5), (1, 0))
    ax2.imshow(medical_img, aspect="auto",cmap='gray')
    ax2.axis('off')

    ax3 = plt.subplot2grid((2, 5), (0, 1))
    img_com = svd_compression(natural_img, k=1)
    ax3.imshow(img_com, aspect="auto",cmap='gray')
    ax3.axis('off')
    ax3.set_title('(b)')#'Spectral norm'
    ax4 = plt.subplot2grid((2, 5), (1, 1))
    img_com = svd_compression(medical_img, k=1)
    ax4.imshow(img_com, aspect="auto",cmap='gray')
    ax4.axis('off')

    ax5 = plt.subplot2grid((2, 5), (0, 2))
    img_com = svd_compression(natural_img, k=30)
    ax5.imshow(img_com, aspect="auto",cmap='gray')
    ax5.axis('off')
    ax5.set_title('(c)') #'Singular values'
    ax6 = plt.subplot2grid((2, 5), (1, 2))
    img_com = svd_compression(medical_img, k=30)
    ax6.imshow(img_com, aspect="auto",cmap='gray')
    ax6.axis('off')

    ax7 = plt.subplot2grid((2, 5), (0, 3))
    _, Sigma, _ = np.linalg.svd(natural_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    cum_sum = np.cumsum(var_sigma)[0:50]
    ax7.plot(np.arange(len(cum_sum)), cum_sum,'g-')
    non_zero_point = var_sigma[np.nonzero(var_sigma)]
    idx = len(non_zero_point)
    ax7.plot(np.arange(len(cum_sum))[idx], cum_sum[idx],'ro')
    ax7.axhline(y=cum_sum[idx], xmin=0, xmax=len(non_zero_point)/len(cum_sum), color='b', linestyle='--')
    ax7.axvline(x=np.arange(len(cum_sum))[idx], ymin=0, ymax=cum_sum[idx], color='b', linestyle='--')
    ax7.set_ylabel('Explained variance')
    ax7.set_title('(d)')
    ax7.grid(b=True, ls=':')
    ax8 = plt.subplot2grid((2, 5), (1, 3))
    _, Sigma, _ = np.linalg.svd(medical_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    cum_sum = np.cumsum(var_sigma)[0:50]
    ax8.plot(np.arange(len(cum_sum)), cum_sum,'g-')
    non_zero_point = var_sigma[np.nonzero(var_sigma)]
    idx = len(non_zero_point)
    ax8.plot(np.arange(len(cum_sum))[idx], cum_sum[idx],'ro')
    ax8.axhline(y=cum_sum[idx], xmin=0, xmax=len(non_zero_point)/len(cum_sum), color='b', linestyle='--')
    ax8.axvline(x=np.arange(len(cum_sum))[idx], ymin=0, ymax=cum_sum[idx], color='b', linestyle='--')
    ax8.set_ylabel('Explained variance')
    ax8.grid(b=True, ls=':')

    ax9 = plt.subplot2grid((2, 5), (0, 4), rowspan=2)
    nat_sd = [0.92, 0.86, 0.86, 0.84, 0.81]
    med_sd = [0.98, 0.94, 0.93, 0.93, 0.90]
    x_axis = [1,2,3,4,5]
    ax9.plot(x_axis, np.array(nat_sd),'go-',label='Natural images')
    ax9.plot(x_axis, np.array(med_sd),'b^-',label='Medical images')
    ax9.set_xticks(x_axis)
    ax9.set_xticklabels(['2', '4', '8', '16', '32'])
    #ax9.set_ylabel('Spectral norm')
    #ax9.set_xlabel('Batch size')
    #ax9.set_title('Explained variance ratio')
    ax9.set_title('(e)')
    ax9.grid(b=True, ls=':')
    ax9.legend()

    #save
    plt.tight_layout()
    plt.savefig('/data/pycode/SFSAttention/imgs/svd.png', dpi=300, bbox_inches='tight')

def plot_svd_shift():
    NATURAL_IMG_PATH = '/data/fjsdata/ImageNet/ILSVRC2012_data/val/n02129165/ILSVRC2012_val_00003788.JPEG'
    MEDICAL_IMG_PATH = '/data/fjsdata/Vin-CXR/train_val_jpg/afb6230703512afc370f236e8fe98806.jpeg'
    natural_img = cv2.imread(NATURAL_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    medical_img = cv2.imread(MEDICAL_IMG_PATH, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(2,3, constrained_layout=True, figsize=(12,8))#

    #column 1 
    #row 1: natural image
    axes[0,0].imshow(natural_img, aspect="auto",cmap='gray')
    axes[0,0].axis('off')
    axes[0,0].set_title('(a)')#Natural Image (280, 415)
    #row 2: medical image
    axes[1,0].imshow(medical_img, aspect="auto",cmap='gray')
    axes[1,0].axis('off')

    #column 2
    #row 1: natural image explained variance
    _, Sigma, _ = np.linalg.svd(natural_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    cum_sum = np.cumsum(var_sigma)[0:50]
    axes[0,1].plot(np.arange(len(cum_sum)), cum_sum,'g-')
    non_zero_point = var_sigma[np.nonzero(var_sigma)]
    idx = len(non_zero_point)
    axes[0,1].plot(np.arange(len(cum_sum))[idx], cum_sum[idx],'ro')
    axes[0,1].axhline(y=cum_sum[idx], xmin=0, xmax=len(non_zero_point)/len(cum_sum), color='b', linestyle='--')
    axes[0,1].axvline(x=np.arange(len(cum_sum))[idx], ymin=0, ymax=cum_sum[idx], color='b', linestyle='--')
    axes[0,1].set_title('(b)')
    axes[0,1].grid(b=True, ls=':')
    #axes[0,1].set_ylabel('Explained variance ratio')
    #axes[0,1].set_xlabel('Singular values')
    #row 2: medical iamge explained variance
    _, Sigma, _ = np.linalg.svd(medical_img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    cum_sum = np.cumsum(var_sigma)[0:50]
    axes[1,1].plot(np.arange(len(cum_sum)), cum_sum,'g-')
    non_zero_point = var_sigma[np.nonzero(var_sigma)]
    idx = len(non_zero_point)
    axes[1,1].plot(np.arange(len(cum_sum))[idx], cum_sum[idx],'ro')
    axes[1,1].axhline(y=cum_sum[idx], xmin=0, xmax=len(non_zero_point)/len(cum_sum), color='b', linestyle='--')
    axes[1,1].axvline(x=np.arange(len(cum_sum))[idx], ymin=0, ymax=cum_sum[idx], color='b', linestyle='--')
    axes[1,1].grid(b=True, ls=':')
    #axes[1,1].set_ylabel('Explained variance ratio')
    #axes[1,1].set_xlabel('Singular values')

    #column 3
    #row 1: natural image k=30
    img_com = svd_compression(natural_img, k=30)
    axes[0,2].imshow(img_com, aspect="auto",cmap='gray')
    axes[0,2].axis('off')
    axes[0,2].set_title('(c)') #'Singular values'
    #row 2: medical image k=30
    img_com = svd_compression(medical_img, k=30)
    axes[1,2].imshow(img_com, aspect="auto",cmap='gray')
    axes[1,2].axis('off')

    """
    #column 4
    #row 1: natural image batch=1 spectral norm
    natural_img = cv2.resize(natural_img,(56,56))
    #natural_img = natural_img.reshape(1, 56*56)
    img_com = svd_compression(natural_img, k=1)
    #axes[0,3].imshow(img_com, aspect="auto",cmap='gray')
    #axes[0,3].axis('off')
    #axes[0,3].set_title('(d)')#'Spectral norm'
    x = np.arange(0,56,1)
    y = np.arange(56-1,0-1,-1)
    X,Y = np.meshgrid(x,y)
    axes[0,3].contourf(X,Y,img_com,6,cmap="YlGnBu")
    axes[0,3].set_title('(d)')
    axes[0,3].axis('off')
    #row 2: medical image batch=1 spectral norm
    medical_img = cv2.resize(medical_img,(56,56))
    #medical_img = medical_img.reshape(1, 56*56)
    img_com = svd_compression(medical_img, k=1)
    #axes[1,3].imshow(img_com, aspect="auto",cmap='gray')
    #axes[1,3].axis('off')
    x = np.arange(0,56,1)
    y = np.arange(56-1,0-1,-1)
    X,Y = np.meshgrid(x,y)
    axes[1,3].contourf(X,Y,img_com,6,cmap="YlGnBu")
    axes[1,3].axis('off')
    
    #column 5
    #row 1: natural image batch_svd
    nat_sd = [0.86, 0.84, 0.81, 0.83, 0.79]
    x_axis = [1,2,3,4,5]
    axes[0,4].plot(x_axis, np.array(nat_sd),'go-')
    axes[0,4].set_xticks(x_axis)
    axes[0,4].set_xticklabels(['8', '16', '32', '64', '128'])
    axes[0,4].set_title('(e)')
    axes[0,4].grid(b=True, ls=':')
    #row 2: medical image batch_svd
    med_sd = [0.95, 0.93, 0.93, 0.92, 0.90]
    x_axis = [1,2,3,4,5]
    axes[1,4].plot(x_axis, np.array(med_sd),'go-')
    axes[1,4].set_xticks(x_axis)
    axes[1,4].set_xticklabels(['8', '16', '32', '64', '128'])
    axes[1,4].grid(b=True, ls=':')

    #column 6-10
    NATURAL_ROOT = '/data/fjsdata/ImageNet/ILSVRC2012_data/val/n02129165/' #natural image 
    MEDICAL_ROOT = '/data/fjsdata/Vin-CXR/train_val_jpg/'#medical image
    #calculate singular degree
    bs = [2-1, 4-1, 8-1, 16-1, 32-1] #[8-1, 16-1, 32-1, 64-1, 128-1]
    title =['(f)', '(g)', '(h)', '(i)', '(j)']
    for i in range(len(bs)):
        batch_img = torch.FloatTensor()
        for _, _, fs in os.walk(NATURAL_ROOT):
            for f in fs: #random.sample(fs, bs[i]):
                if batch_img.size(0) == bs[i]: break
                img = os.path.join(NATURAL_ROOT, f)
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(56,56))
                batch_img = torch.cat((batch_img, torch.Tensor(img).unsqueeze(0)), 0)
        img = cv2.imread(NATURAL_IMG_PATH, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(56,56))
        batch_img = torch.cat((batch_img, torch.Tensor(img).unsqueeze(0)), 0)
        batch_img = batch_img.view(batch_img.size(0), -1)
        img_com = svd_compression(batch_img.numpy(), k=1)
        img_com = img_com[-1,:].reshape((56,56))
        #axes[0,5+i].imshow(img_com, aspect="auto",cmap='gray')
        #axes[0,5+i].set_title(title[i])
        #axes[0,5+i].axis('off')
        x = np.arange(0,56,1)
        y = np.arange(56-1,0-1,-1)
        X,Y = np.meshgrid(x,y)
        axes[0,5+i].contourf(X,Y,img_com,6,cmap="YlGnBu")
        axes[0,5+i].set_title(title[i])
        axes[0,5+i].axis('off')

        #four row
        batch_img = torch.FloatTensor()
        for _, _, fs in os.walk(MEDICAL_ROOT):
            for f in fs:
                if batch_img.size(0) == bs[i]: break
                img = os.path.join(MEDICAL_ROOT, f)
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(56,56))
                batch_img = torch.cat((batch_img, torch.Tensor(img).unsqueeze(0)), 0)
        img = cv2.imread(MEDICAL_IMG_PATH, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(56,56))
        batch_img = torch.cat((batch_img, torch.Tensor(img).unsqueeze(0)), 0)
        batch_img = batch_img.view(batch_img.size(0), -1)
        img_com = svd_compression(batch_img.numpy(), k=1)
        img_com = img_com[-1,:].reshape((56,56))
        #axes[1,5+i].imshow(img_com, aspect="auto",cmap='gray')
        #axes[1,5+i].axis('off')
        x = np.arange(0,56,1)
        y = np.arange(56-1,0-1,-1)
        X,Y = np.meshgrid(x,y)
        axes[1,5+i].contourf(X,Y,img_com,6,cmap="YlGnBu")
        axes[1,5+i].axis('off')
    """
    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/img_com.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    """
    W = torch.rand(100,200)
    u, s, v = power_iteration(W)
    print(s)
    U, S, V = torch.svd(W)
    print(S.max())
    """
    #plot_svd_compression()
    #calculate_batch_SN()
    #plot_svd_compression2()
    plot_svd_shift()
