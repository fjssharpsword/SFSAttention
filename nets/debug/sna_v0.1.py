# encoding: utf-8
"""
Spectral Norm Attention.
Author: Jason.Fang
Update time: 16/09/2021
"""
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor


class SNALayer(nn.Module): 
    def __init__(self, channels, Ip=10):
        super(SNALayer, self).__init__()

        self.Ip = Ip
        #spatial-wise
        self.conv = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False)

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
    
    def batch_forward(self, x):

        #reducing redundancy of single feature map with channels.
        B, C, H, W = x.shape
        w = x.view(B, C, H * W).permute(0, 2, 1)  # B * N * C, where N = H*W

        #spectral normalization
        u, v= self._batch_power_iteration(w)
        
        z = torch.bmm(u, v.permute(0, 2, 1))  # B * N * C, where N = H*W
        z = z.permute(0, 2, 1).view(B, C, H, W)  # B * C * H * W

        x = torch.add(z, x) # z + x
        return x

    def _power_iteration(self, W):
        """
        power iteration for max_singular_value
        """
        v = torch.FloatTensor(W.size(1), 1).normal_(0, 1).cuda()
        W_s = torch.matmul(W.T, W)
        for _ in range(self.Ip):
            v_t = v
            v = torch.matmul(W_s, v_t)
            v = v/torch.norm(v)

        u = torch.matmul(W, v)
        u = u/torch.norm(u)
        
        return u, v #left vector, right vector

    def forward(self, x):
        
        #reducing redundancy of batch feature maps without channels.
        B, C, H, W = x.shape
        #spatial-wise
        w_s = self.conv(x) # B * 1 * H * W 

        if self.eval():#test
            u, v= self._batch_power_iteration(w_s.squeeze())# B *H * W
            w = torch.bmm(u, v.permute(0, 2, 1)).unsqueeze(1)  # B * 1* H * W
            x = torch.add(w, x) 
            return x

        #SVD for reducing redundancy of features
        w_s = w_s.squeeze().view(B, H*W) #B * N, where N= H *W 
        u, v = self._power_iteration(w_s)
        #calculate the attentive score
        w_s = torch.matmul(u, v.T).view(B, 1, H, W) # B * 1 * H  * W
        #redisual addition
        x = torch.add(x, w_s)

        return x
       
        """
        #reducing redundancy of batch feature maps with channels.
        B, C, H, W = x.shape
        #spatial-wise
        #SVD for reducing redundancy of features
        w_s = x.view(B*C, H*W) #B*C, N, where N= H *W 
        u, v = self._power_iteration(w_s)
        #calculate the attentive score
        w_s = torch.matmul(u, v.T).view(B, C, H, W) # B * C* H  * W
        #redisual addition
        x = torch.add(x, w_s)

        return x
        """

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(8, 512, 32, 32).cuda()
    ssa = SNALayer(channels=512).cuda()
    start = time.time()
    out = ssa(x)
    end = time.time()
    time_elapsed = end - start
    print(time_elapsed)
    print(out.shape)