'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))

    def INF(self, B, H, W, device):
        # Modified to work on both CPU and GPU
        diag = torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0)
        return -diag.unsqueeze(0).repeat(B*W, 1, 1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        device = x.device  # Get the device of the input tensor
        
        # Project queries, keys and values
        proj_query = self.query_conv(x)  # B x C' x H x W
        proj_key = self.key_conv(x)      # B x C' x H x W
        proj_value = self.value_conv(x)  # B x C x H x W
        
        # For height dimension attention
        # Reshape to B*W x C' x H
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        # Reshape to B*W x H x C'
        proj_key_H = proj_key.permute(0, 3, 2, 1).contiguous().view(m_batchsize*width, height, -1)
        
        # For width dimension attention
        # Reshape to B*H x C' x W
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)
        # Reshape to B*H x W x C'
        proj_key_W = proj_key.permute(0, 2, 3, 1).contiguous().view(m_batchsize*height, width, -1)
        
        # Value projections reshape
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize*width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize*height, -1, width)
        
        # Calculate attention maps
        # B*W x H x H
        energy_H = torch.bmm(proj_key_H, proj_query_H)
        energy_H = energy_H + self.INF(m_batchsize, height, width, device)
        energy_H = energy_H.view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        
        # B*H x W x W
        energy_W = torch.bmm(proj_key_W, proj_query_W)
        energy_W = energy_W.view(m_batchsize, height, width, width)
        
        # Apply softmax to get attention weights
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        
        # Split attention maps
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize*width, height, height)
        att_W = concate[:, :, :, height:height+width].contiguous().view(m_batchsize*height, width, width)
        
        # Apply attention maps to values
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        
        # Combine the outputs and apply residual connection
        output = out_H + out_W
        output = self.gamma * output + x
        
        return output



if __name__ == '__main__':
    model = CrissCrossAttention(64)
    x = torch.randn(2, 64, 5, 6)
    out = model(x)
    print(out.shape)
