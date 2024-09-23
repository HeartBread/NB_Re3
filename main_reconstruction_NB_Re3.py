import sys
import time

import gc
import math
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm, trange

from data_loader_NB_Re3 import get_train_loader, get_test_loader
from models import NB_Re3
from support.utils import arr2raster, read_images, filelist_filter


# 重定向终端或控制台的输出内容
class logger(object):
    def __init__(self, log, stream=sys.stdout):
        self.terminal = stream
        self.log = log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def arr2dict(data, device):
    srs_data = data[0]
    masks_data = data[1]
    lstm_intervals = data[2]
    lstm_intervals_inver = data[3]
    ndvis_data = data[4]

    srs_data_inver = torch.flip(srs_data, dims=[2])
    masks_data_inver = torch.flip(masks_data, dims=[2])
    ndvis_data_inver = torch.flip(ndvis_data, dims=[2])

    forward_data = {'srs_data': srs_data.to(device).requires_grad_(),
                    'masks_data': masks_data.to(device).requires_grad_(),
                    'lstm_intervals': lstm_intervals.to(device).requires_grad_(),
                    'ndvis_data': ndvis_data.to(device).requires_grad_()}

    backward_data = {'srs_data': srs_data_inver.to(device).requires_grad_(),
                     'masks_data': masks_data_inver.to(device).requires_grad_(),
                     'lstm_intervals': lstm_intervals_inver.to(device).requires_grad_(),
                     'ndvis_data': ndvis_data_inver.to(device).requires_grad_()}

    ret_input = {'forward': forward_data, 'backward': backward_data}

    return ret_input


def train(model, npy_data_path, train_prefix, epochs, batch_size, device, loss_log):
    # 训练过程
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    tp_1 = time.perf_counter()
    train_data_iter = get_train_loader(batch_size=batch_size, npy_path=npy_data_path, prefix=train_prefix, shuffle=True)
    tp_2 = time.perf_counter()
    print(' Training data loading consume: %s min' % ((tp_2 - tp_1) / 60))

    print('train process (epochs):')

    with tqdm(range(epochs), ncols=100) as pbar:
        
        for epoch in pbar:
            model.train()
            run_loss = 0.0

            for idx, data in enumerate(train_data_iter):
                data = arr2dict(data, device)
                ret = model.run_on_batch(data, optimizer)
                run_loss += ret['loss'].data.item()
                print(' Progress epoch {}, {:.2f}%, average loss {:.10f}'.format(epoch, (idx + 1) * 100.0 / len(train_data_iter), run_loss / (idx + 1.0)), file=loss_log)

            print('===============================================================', file=loss_log)

            scheduler.step()

        last_bar = str(pbar)
        sys.stdout.write(last_bar + '\r')

    print('train process end.')
    print('===============================================================')


def evaluate(model, npy_data_path, test_prefix, seq_len, feat_num, batch_size, device):
    
    with torch.no_grad():
        
        model.eval()

        tp_1 = time.perf_counter()
        eval_data_iter = get_test_loader(batch_size=batch_size, npy_path=npy_data_path, prefix=test_prefix, shuffle=False)
        tp_2 = time.perf_counter()
        print(' Testing data loading consume: %s min' % ((tp_2 - tp_1) / 60))

        length = len(eval_data_iter.sampler)
        impute = np.zeros((length, feat_num, seq_len), dtype=np.float32)
        current_pix_num = 0
        batch_num = math.ceil(length / batch_size)
        print('valid process (batch samplers):')

        with tqdm(enumerate(eval_data_iter), total=batch_num, ncols=100) as pbar:
            
            for idx, data in pbar:
                
                data = arr2dict(data, device)
                ret = model.run_on_batch(data, None)

                # format of imputation: [batch_size, feat_num, seq_len]
                imputation = ret['imputations'].data.cpu().numpy()
                batch_size = imputation.shape[0]

                impute[current_pix_num: current_pix_num + batch_size, :, :] = imputation  # only model imputations
                current_pix_num += batch_size

            last_bar = str(pbar)
            sys.stdout.write(last_bar + '\r')
    
    np.save(npy_path + '/Reconstructed_reflectance', impute)
    tp_3 = time.perf_counter()
    print(' validate consume: %s min' % ((tp_3 - tp_2) / 60))
    print('validate process end.')
    print('===============================================================')
    

def batch_output(npy_path):
    
    output_batch = npy_path + '/Batch_reflectance'
    imputed_ref = np.load(npy_path + '/Reconstructed_reflectance.npy').astype(np.float32)
    
    with tqdm(range(seq_len), ncols=100) as pbar:
        for i_image in pbar:
            ref_i = imputed_ref[:, :, i_image]
            np.save(output_batch + '/Reconstructed_reflectance_' + str(i_image + 1) + '.npy', ref_i)
            del ref_i
            gc.collect()
            
        last_bar = str(pbar)
        sys.stdout.write(last_bar + '\r')
    
    print('===============================================================')
    
def output(npy_path, output_data_path, seq_len, feat_num, width, height):
    
    tp_1 = time.perf_counter()
    roi_index = np.load(npy_path + '/Roi_index.npy')
    tp_2 = time.perf_counter()
    print(' input data: %s min' % ((tp_2 - tp_1) / 60))
    
    output_batch = npy_path + '/Batch_reflectance'
    save_impute = np.zeros((feat_num * seq_len, height, width), dtype=np.float32)
    with tqdm(range(seq_len), ncols=100) as pbar:
        for i_image in pbar:
            save_impute_i = np.zeros((width * height, feat_num, 1), dtype=np.float32)
            imputed_ref_i = np.load(output_batch + '/Reconstructed_reflectance_' + str(i_image + 1) + '.npy').astype(np.float32)
            save_impute_i[roi_index, :, 0] = imputed_ref_i
            save_impute_i = np.transpose(save_impute_i, (2, 1, 0)).reshape((feat_num * 1, width * height))
            save_impute_i = save_impute_i.reshape((-1, width, height)).transpose((0, 2, 1))
            
            save_impute[feat_num * i_image : feat_num * (i_image + 1), :, :] = save_impute_i
            del save_impute_i, imputed_ref_i
            gc.collect()
        
        last_bar = str(pbar)
        sys.stdout.write(last_bar + '\r')
    
    dates_images = dates_file.read().splitlines()

    dates_images = [f'{i+1}_{image}' for i, image in enumerate(dates_images)]
    dates_images_repeat_data = []
    for i, image in enumerate(dates_images):
        for j in range(feat_num):
            new_image_data = f'{image}_({band_name_data[j]})'
            dates_images_repeat_data.append(new_image_data)

    dates_file.close()
    
    output_raster = output_data_path + '/' + Study_Area + '_Reconstructed_2022.tif'
    
    arr2raster(save_impute, output_raster, band_names=dates_images_repeat_data)
    print(' successfully output the imputed raster.')
    
    tp_2 = time.perf_counter()
    print(' reconstructed time-series output consume: %s min' % ((tp_2 - tp_1) / 60))
    print('output process end.')
    print('===============================================================')


# main function
if __name__ == '__main__':

    t1 = time.perf_counter()

    epochs = 10  # 200
    train_batch_size = 2048  # 256
    evaluate_batch_size = 2048

    # 参数修改 ====================================================
    seq_len = 49
    feat_num = 10
    drop_out = 0.2

    width = 6031
    height = 6201

    date_reference = '2022-03-01'  # '2019-09-30', '2022-03-01'
    
    train_prefix = 'roi'  # 是否采样训练数据, 需要修改, 'sample' or 'all' or 'no_NAN'
    test_prefix = 'roi'
    
    Area_Name = 'Mangkang_Changdu'
    Study_Area = 'Mangkang_Changdu'
    fine_directory = '/data/newhome/hongtao' + '/' + Study_Area
    
    npy_path = fine_directory + '/npy_data'
    dates_path = fine_directory
    output_path = fine_directory + '/output_temp'
    
    dates_file = open(fine_directory + '/dates_timeseries.txt')
    band_name_data = ['2_blue', '3_green', '4_red', '5_red_edge_1', '6_red_edge_2', '7_red_edge_3', '8_nir', '8A_red_edge_4', '11_swir_1', '12_swir_2']
    
    # 参数修改 ====================================================
    tcn_kernel_size = 3
    tcn_num_channels = [64, 128, 256]  # [16, 32, 64, 128, 256, 512, 1024], [total_feat_num] * 3, [16, 32, 64]

    lstm_hidden_size = 96

    is_train = False
    is_validate = True
    batch_out = True
    is_output = True

    # 重定向终端或控制台的输出内容
    terminal_log = open(output_path + '/time_cost.txt', 'w')
    sys.stdout = logger(terminal_log)

    if is_train:

        print('Training ...')

        # 插值网络使用影像全部样本作为输入边训练边评价, 不用区分训练集和测试集
        model_train = NB_Re3.NB_Re3(seq_len, feat_num, tcn_kernel_size, tcn_num_channels, lstm_hidden_size, drop_out)

        params = sum(p.numel() for p in model_train.parameters() if p.requires_grad)
        print(' Model params is {}.'.format(params))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_train = model_train.to(device)
 
        loss_log = open(output_path + '/loss_log.txt', 'w')

        print('Training:', file=loss_log)

        train(model_train, npy_path, train_prefix, epochs, train_batch_size, device, loss_log)  # train model
        torch.save(model_train.state_dict(), output_path + '/model_para.pth')  # save training parameters

        loss_log.close()

    if is_validate:

        print('Testing ...')

        # 插值网络使用影像全部样本作为输入边训练边评价, 不用区分训练集和测试集
        model_evaluate = NB_Re3.NB_Re3(seq_len, feat_num, tcn_kernel_size, tcn_num_channels, lstm_hidden_size, drop_out)

        params = sum(p.numel() for p in model_evaluate.parameters() if p.requires_grad)
        print(' Model params is {}.'.format(params))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_evaluate = model_evaluate.to(device)

        model_evaluate.load_state_dict(torch.load(output_path + '/model_para.pth'))

        # evaluate model and output the imputed raster
        evaluate(model_evaluate, npy_path, test_prefix, seq_len, feat_num, evaluate_batch_size, device)
    
    if batch_out:
        
        print('Batch the output npy ...')
        batch_output(npy_path)
        
    if is_output:
        
        print('Outputting ...')
        output(npy_path, output_path, seq_len, feat_num, width, height)

    t2 = time.perf_counter()
    print('模型总运行时间: %s min' % ((t2 - t1) / 60))
