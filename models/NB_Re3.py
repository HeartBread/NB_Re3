import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

from models.blocks.temporal_conv import TemporalConvNet
from models.blocks.temporal_decay import TemporalMissing


class WeightedSum(nn.Module):

    def __init__(self, batch_size, seq_len, feat_num):

        super(WeightedSum, self).__init__()

        if torch.cuda.is_available():
            self.weights_res = Parameter(torch.zeros((batch_size * seq_len, feat_num)).cuda(), requires_grad=True)
        else:
            self.weights_res = Parameter(torch.zeros((batch_size * seq_len, feat_num)), requires_grad=True)

    def forward(self, input_a, input_b):

        weighted_sum = (0.5 + self.weights_res) * input_a + (0.5 - self.weights_res) * input_b

        return weighted_sum


# ******************************************************************************************************
# **********************************************Main*Block**********************************************

# one-directional LSTM Model building
# tcn_num_channels: number of output channels from TCN to lstm
# tcn_k_size: kernel_size of TCN
# drop_out: drop_out ratio of TCN
# lstm_hidden_size: number of features in the lstm hidden state `h`
# seq_len: length of sequence
# feat_num: band number of reflectance (feature number)

class N_Re3(nn.Module):
    
    def __init__(self, seq_len, feat_num, tcn_k_size, tcn_num_channels, lstm_hidden_size, drop_out):

        super(N_Re3, self).__init__()

        self.seq_len = seq_len
        self.feat_num = feat_num
        self.drop_out = drop_out
        self.tcn_kernel_size = tcn_k_size
        self.lstm_hidden_size = lstm_hidden_size
        self.tcn_num_channels = tcn_num_channels

        # 1d-temporal-convolution for ndvi trend, tcn layer
        self.tcn_ndvi = TemporalConvNet(num_inputs=self.feat_num, num_channels=self.tcn_num_channels, kernel_size=self.tcn_kernel_size, dropout=self.drop_out)

        # time-stepped-lstm for accumulative temporal-missing/decay patterns
        self.temp_miss_lstm_1 = TemporalMissing(input_size=self.feat_num, output_size=self.lstm_hidden_size, diag=False)

        self.lstm_cell_1 = nn.LSTMCell(input_size=self.tcn_num_channels[-1] + self.feat_num, hidden_size=self.lstm_hidden_size)

        # full-connection layer
        self.linear_reg_lstm = nn.Linear(in_features=self.lstm_hidden_size, out_features=self.feat_num)
        # Multi-band correlation
        self.linear_reg_ref = nn.Linear(in_features=self.feat_num, out_features=self.feat_num)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    # srs_seq: [batch_size, feat_num, seq_len]
    # srs_masked_seq: [batch_size, feat_num, seq_len, k_size], selected SRef for each timestep.
    # ndvis_seq: [batch_size, 1 * feat_num, seq_len]
    # masks_seq: [batch_size, 1 * feat_num, seq_len], value 0 for clear and value 1 for cloud
    # intervals_masked_seq: [batch_size, feat_num, seq_len, k_size], selected sorted intervals for each timestep.
    # lstm_intervals_seq: [batch_size, seq_len, feat_num], temporal decay intervals for lstm
    def forward(self, data, direct='forward'):

        srs_seq = data[direct]['srs_data']
        masks_unique_seq = data[direct]['masks_data']
        lstm_intervals_unique_seq = data[direct]['lstm_intervals']
        ndvis_unique_seq = data[direct]['ndvis_data']

        b_size = srs_seq.shape[0]

        masks_seq = masks_unique_seq.repeat(1, self.feat_num, 1)
        ndvis_seq = ndvis_unique_seq.repeat(1, self.feat_num, 1)
        lstm_intervals_seq = lstm_intervals_unique_seq.repeat(1, self.feat_num, 1)

        # ndvi process
        tcn_ndvi_output = self.tcn_ndvi(ndvis_seq)  # [batch_size, feat_num, seq_len] ----------→ [batch_size, num_channels[-1], seq_len]

        # h, c of shape '(batch, hidden_size)'
        if torch.cuda.is_available():
            h1 = Variable(torch.zeros((b_size, self.lstm_hidden_size)).cuda())
            c1 = Variable(torch.zeros((b_size, self.lstm_hidden_size)).cuda())
        else:
            h1 = Variable(torch.zeros((b_size, self.lstm_hidden_size)))
            c1 = Variable(torch.zeros((b_size, self.lstm_hidden_size)))

        x_loss_lstm = 0.0
        imputations_lstm = []

        # lstm: 逐timestep计算
        for t in range(self.seq_len):

            sr_y = srs_seq[:, :, t]
            ndvi_tcn_y = tcn_ndvi_output[:, :, t]
            mask_y = masks_seq[:, :, t]
            delta_y = lstm_intervals_seq[:, :, t]

            gamma_h_1 = self.temp_miss_lstm_1(delta_y)

            h1 = h1 * gamma_h_1

            # format of sr_h: [bsize, feat_num]
            sr_h = self.linear_reg_lstm(h1)  # 由上一时刻的隐藏状态得到这一时刻的band预测输入
            x_loss_lstm += torch.sum((torch.abs(sr_y - sr_h)) * (1 - mask_y)) / (torch.sum((1 - mask_y)) + 1e-5)  # 重构误差

            sr_c = sr_y * (1 - mask_y) + mask_y * sr_h  # 判断当前时刻是否缺失: not observed, sr_h; observed, sr_y

            inputs_1 = torch.cat([ndvi_tcn_y, sr_c], dim=1)

            # update (h, c) at this time step with (h, c) at last time step
            if torch.cuda.is_available():
                inputs_1 = inputs_1.cuda()
            h1, c1 = self.lstm_cell_1(inputs_1, (h1, c1))
            
            sr_z = self.linear_reg_ref(sr_c)
            x_loss_lstm += torch.sum((torch.abs(sr_y - sr_z)) * (1 - mask_y)) / (torch.sum((1 - mask_y)) + 1e-5)  # 重构误差

            imputations_lstm.append(sr_z.unsqueeze(dim=2))

        imputations_lstm_eval = torch.cat(imputations_lstm, dim=2)
        x_loss_lstm = 1.0 * (x_loss_lstm / self.seq_len)

        # output format: [batch_size, feat_num, seq_len]
        return {'loss': x_loss_lstm, 'imputations': imputations_lstm_eval, 'true_values': srs_seq, 'masks': masks_seq}

    def run_on_batch(self, data, optimizer):

        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret


# Bidirectional LSTM Model building
class NB_Re3(nn.Module):

    def __init__(self, seq_len, feat_num, tcn_k_size, tcn_num_channels, lstm_hidden_size, drop_out):

        super(NB_Re3, self).__init__()

        self.seq_len = seq_len
        self.feat_num = feat_num
        self.drop_out = drop_out
        self.lstm_hidden_size = lstm_hidden_size
        self.tcn_kernel_size = tcn_k_size
        self.tcn_num_channels = tcn_num_channels

        self.lits_f = N_Re3(seq_len=self.seq_len, feat_num=self.feat_num, tcn_k_size=self.tcn_kernel_size, tcn_num_channels=self.tcn_num_channels, lstm_hidden_size=self.lstm_hidden_size, drop_out=self.drop_out)
        self.lits_b = N_Re3(seq_len=self.seq_len, feat_num=self.feat_num, tcn_k_size=self.tcn_kernel_size, tcn_num_channels=self.tcn_num_channels, lstm_hidden_size=self.lstm_hidden_size, drop_out=self.drop_out)

    def forward(self, data):

        ret_f = self.lits_f(data, direct='forward')
        ret_b = self.reverse(self.lits_b(data, direct='backward'))
        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):

        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss = loss_f + loss_b

        b_size = ret_f['true_values'].shape[0]
        weighted_sum = WeightedSum(b_size, self.seq_len, self.feat_num)

        imputations_f = ret_f['imputations'].permute(0, 2, 1).contiguous().view(-1, self.feat_num)
        imputations_b = ret_b['imputations'].permute(0, 2, 1).contiguous().view(-1, self.feat_num)
        ret_c = weighted_sum(imputations_f, imputations_b)
        ret_c = ret_c.view(-1, 1)

        # weighted loss
        true_values = ret_f['true_values'].permute(0, 2, 1).contiguous().view(-1, 1)
        masks = ret_f['masks'].permute(0, 2, 1).contiguous().view(-1, 1)

        loss_w = torch.sum((torch.exp(torch.abs(ret_c - true_values)) - 1) * (1 - masks)) / (torch.sum(1 - masks) + 1e-5)
        ret_f['imputations'] = ret_c.reshape((-1, self.seq_len, self.feat_num)).permute(0, 2, 1)

        loss += loss_w
        ret_f['loss'] = loss

        return ret_f

    def reverse(self, ret):

        def reverse_tensor(tensor_):

            if tensor_.dim() <= 1:
                return tensor_

            indices = range(tensor_.size()[2])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            if torch.cuda.is_available():
                indices = Variable(torch.LongTensor(indices).cuda(), requires_grad=False)
            else:
                indices = Variable(torch.LongTensor(indices), requires_grad=False)

            return tensor_.index_select(2, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer):

        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
