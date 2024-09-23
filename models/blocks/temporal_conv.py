# TCN Network
# TCN核心模块(TemporalBlock), 包含2个Conv1d操作和1个残差连接部分
# n_inputs: int, 输入通道数或特征数
# n_outputs: int, 输出通道数或特征数
# kernel_size: int, 卷积核尺寸
# stride: int, 步长, 在TCN固定为1
# dilation: int, 膨胀系数, 与Block所在的层数有关系, dilation_size = 2 ** i
# padding: int, 填充系数, 与kernel_size和dilation有关
# dropout: float, dropout比例

import torch.nn as nn
from torch.nn.utils import weight_norm
from models.blocks.partial_conv import PartialConv1d


class TemporalBlock(nn.Module):

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()

        # 输出size: [batch_size, input_channel, padding + seq_len + padding]
        self.conv1 = weight_norm(PartialConv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, multi_channel=True, bias=True))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(PartialConv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation, multi_channel=True, bias=True))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)

        # 1×1的卷积, 只有在进入Block的通道数和输出Block的通道数不一样时使用. 一般都会不一样, 除非num_channels里面的数与num_inputs相等.
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None

        # 非线性激活函数
        self.relu = nn.ReLU()

        # 参数初始化
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # size of x: [batch_size, input_channel, seq_len]
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)


# num_inputs: int, 输入通道数或特征数
# num_channels: list, 每层输出的hidden_channel数. Example: [5, 12, 3], 3 blocks: (block_1: 5; block_2: 12; block_3: 3)
# kernel_size: int, 卷积核尺寸
# dropout: float, dropout比例
class TemporalConvNet(nn.Module):

    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)  # 网络深度 N

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 2) * dilation_size, dropout=dropout)]
            # origin TCN for casual convolution
            # layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    # size of x: [batch_size, input_channel, seq_len]
    # return: [batch_size, output_channel, seq_len]
    def forward(self, x):
        return self.network(x)
