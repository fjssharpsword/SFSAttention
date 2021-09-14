# encoding: utf-8
"""
Spectral Norm for Self-attention.
Author: Jason.Fang
Update time: 15/09/2021
"""
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class SSALayer(nn.Module): 
    def __init__(self, in_ch, k, k_size=3):
        super(SSALayer, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        self.f = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.g = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

        for conv in [self.f, self.g, self.h]: 
            conv.apply(weights_init)
        self.v.apply(constant_init)

    #approximated SVD
    #https://jeremykun.com/2016/05/16/singular-value-decomposition-part-2-theorem-proof-algorithm/
    def _power_iteration(self, W, eps=1e-10):
        """
        power iteration for max_singular_value
        """
        v = torch.FloatTensor(W.size(1), 1).normal_(0, 1).cuda()
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
        
        return u, s, v #left vector, sigma, right vector

    def forward(self, x):

        B, C, H, W = x.shape

        f_x = self.f(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W

        attn = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * N * N, where N = H*W
        #spectral normalizaiton
        
        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W)  # B * mid_ch * H * W

        z = self.v(z)
        x = torch.add(z, x) # z + x
        return x


## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(8, 512, 32, 32).cuda()
    ssa = SSALayer(in_ch=512, k=2, k_size=5).cuda()
    start = time.time()
    out = ssa(x)
    end = time.time()
    time_elapsed = end - start
    print(time_elapsed)
    print(out.shape)