# Temporal missing pattern decay
# Because the time series may be irregularly sampled because of random cloud contamination,
# we used the temporal decay factor 'gamma' to represent the missing patterns in the time series
# gamma_t = exp{-max(0, W_gamma * delta_t+ b_gamma)}

import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
from torch.nn import Parameter


class TemporalMissing(nn.Module):

    def __init__(self, input_size, output_size, diag=False):

        super(TemporalMissing, self).__init__()

        self.diag = diag

        self.W = Parameter(torch.Tensor(output_size, input_size), requires_grad=True)
        self.b = Parameter(torch.Tensor(output_size), requires_grad=True)

        if self.diag:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)  # 生成对角线全是1，其余部分全是0的二维数组
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)

        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):

        if self.diag:
            gamma = f.relu(f.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = f.relu(f.linear(d, self.W, self.b))
            
        gamma = torch.exp(-gamma)

        return gamma
    
