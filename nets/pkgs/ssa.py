# encoding: utf-8
"""
Spectral Spatial-Attention.
Author: Jason.Fang
Update time: 16/09/2021
"""
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class SSALayer(nn.Module): 
    def __init__(self, Ip=10):
        super(SSALayer, self).__init__()
        self.Ip = Ip

        #spatial-wise
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        #channel-wise
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def _batch_power_iteration(self, W):
        """
        power iteration for max_singular_value
        """
        v = torch.FloatTensor(W.size(0), W.size(2), 1).normal_(0, 1).cuda()
        W_s = torch.bmm(W.permute(0, 2, 1), W)
        for _ in range(self.Ip):
            v_t = v
            v = torch.bmm(W_s, v_t)
            v_norm = torch.norm(v.squeeze(), dim=1).unsqueeze(-1).unsqueeze(-1)
            v_norm = v_norm.expand_as(v)
            v = torch.div(v, v_norm)

        u = torch.bmm(W, v)
        u_norm = torch.norm(u.squeeze(), dim=1).unsqueeze(-1).unsqueeze(-1)
        u_norm = u_norm.expand_as(u)
        u = torch.div(u, u_norm)
        return u, v #left vector, right vector

    def forward(self, x):

        B, C, H, W = x.shape
        w = x.view(B, C, H * W).permute(0, 2, 1)  # B * N * C, where N = H*W

        #spectral normalization
        u, v= self._batch_power_iteration(w)
        
        z = torch.bmm(u, v.permute(0, 2, 1))  # B * N * C, where N = H*W
        z = z.permute(0, 2, 1).view(B, C, H, W)  # B * C * H * W

        x = torch.add(z, x) # z + x
        return x

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(8, 512, 32, 32).cuda()
    ssa = SSALayer().cuda()
    start = time.time()
    out = ssa(x)
    end = time.time()
    time_elapsed = end - start
    print(time_elapsed)
    print(out.shape)