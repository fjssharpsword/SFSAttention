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
    IMG_PATH = '/data/fjsdata/ImageNet/ILSVRC2012_data/val/n02129165/ILSVRC2012_val_00003788.JPEG'
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    fig, axes = plt.subplots(1,3, constrained_layout=True, figsize=(9,3))

    #origin image
    axes[0].imshow(img, aspect="auto",cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Origin Image')
    #k=1
    img_com = svd_compression(img, k=1)
    axes[1].imshow(img_com, aspect="auto",cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Rank 1')
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
    _, Sigma, _ = np.linalg.svd(img)
    var_sigma = np.round(Sigma**2/np.sum(Sigma**2), decimals=3)
    var_sigma = var_sigma[np.nonzero(var_sigma)]
    sns.barplot(x=list(range(1,len(var_sigma)+1)), y=var_sigma, color="limegreen", ax =axes[2] )
    axes[2].set_ylabel('Explained variance (%)')
    axes[2].set_xlabel('Number of non-zero SVs')
    for ind, label in enumerate(axes[2].xaxis.get_ticklabels()):
        if ind == 0: label.set_visible(True)
        elif (ind+1) % 8 == 0:   # every 4th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    #save
    fig.savefig('/data/pycode/SFSAttention/imgs/svd_com.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    """
    W = torch.rand(100,200)
    u, s, v = power_iteration(W)
    print(s)
    U, S, V = torch.svd(W)
    print(S.max())
    """
    plot_svd_compression()