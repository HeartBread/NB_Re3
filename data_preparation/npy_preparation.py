# generate npy data
import time
import sys
import numpy as np
from tqdm import tqdm, trange
import math
import gc
from support.utils import filelist_filter, read_images, interval_count

sys.path.append('/data/newhome/hongtao/Pyproject/Reflectance_Reconstruction')

t1 = time.perf_counter()

# ******************************************************************************************************
study_area = 'Area_1'
data_directory = '/data/newhome/hongtao/Area_1'
raw_directory = data_directory + '/raw_image'
mask_directory = data_directory + '/image_mask'
img_directory = data_directory + '/image_data'
output_directory = data_directory + '/npy_data'

################################

m, n = 3, 2  # height, width
idx = 0

block_size = 3
is_train = False
is_block = False  # only set true for training data
is_split = True  # only set true when split proecss needed
is_dismatch = False  # only set true when dimension dismatch
################################

file_filter_data = filelist_filter(raw_directory, prefix_label='S2_' + study_area + '_', suffix_label = '.tif')
file_filter_qa = filelist_filter(mask_directory, prefix_label='S2_' + study_area + '_qa_', suffix_label = '.tif')
file_filter_cp = filelist_filter(mask_directory, prefix_label='S2_' + study_area + '_Cloud_probability_', suffix_label = '.tif')
file_filter_cs = filelist_filter(mask_directory, prefix_label='S2_' + study_area + '_Cloud_score_', suffix_label = '.tif')

dates_file = open(data_directory + '/dates_timeseries.txt')
date_reference = '2022-03-01'

sr_bands = 10
scale_factor = 0.0001
band_name_data = ['2_blue', '3_green', '4_red', '5_red_edge_1', '6_red_edge_2', '7_red_edge_3', '8_nir', '8A_red_edge_4', '11_swir_1', '12_swir_2']
# ******************************************************************************************************

################################################################
""" s2 cloud detection function based on QA60 band """
# input: pixel_qa, output: cloud (true), clear (false)
def cloud_s2(data):
    data = data.astype(np.int16)
    return np.bitwise_and(np.right_shift(data, 10), 11) != 0

""" temporal decay intervals calculation """
def lstm_intervals(npy_path, cloud_data, dates_file, date_reference, seq_len):
    
    cloud_data_inver = np.flip(cloud_data, axis=2)
    
    dates_images = dates_file.read().splitlines()
    dates_file.close()
    
    intervals_count = np.vectorize(interval_count)  # 将 interval_count 函数向量化
    date_intervals = intervals_count(dates_images, date_reference)
    date_intervals_inver = np.flip(date_intervals)
    
    mask_delta = np.zeros(cloud_data.shape).astype(np.int16)
    mask_delta_inver = np.zeros(cloud_data_inver.shape).astype(np.int16)
    
    for t in trange(seq_len):
        if t == 0:
            mask_delta[:, :, t] = 0
            mask_delta_inver[:, :, t] = 0
        else:
            mask_delta[:, :, t] = (date_intervals[t] - date_intervals[t - 1]) + mask_delta[:, :, t - 1] * cloud_data[:, :, t - 1]
            mask_delta_inver[:, :, t] = (date_intervals_inver[t - 1] - date_intervals_inver[t]) + mask_delta_inver[:, :, t - 1] * cloud_data_inver[:, :, t - 1]
            
    np.save(npy_path + '/LSTM_intervals_train_roi', mask_delta)
    np.save(npy_path + '/LSTM_intervals_train_inver_roi', mask_delta_inver)
    del mask_delta, mask_delta_inver
    
    t2 = time.perf_counter()
    print(' lstm_intervals: %s s' % (t2 - t1))


def spatial_sample(height, width, block_size):
    
    # 空间均匀采样
    num_blocks_row = (height + 1) // block_size
    num_blocks_col = (width + 1) // block_size
    center_indices = np.zeros((num_blocks_row * num_blocks_col, 2)).astype(np.int16)
    n_b = 0

    # 遍历行
    for i in range(num_blocks_col):

        # 计算行的起始和结束索引
        row_start = i * block_size
        row_end = (i + 1) * block_size

        # 遍历列
        for j in range(num_blocks_row):
            # 计算列的起始和结束索引
            col_start = j * block_size
            col_end = (j + 1) * block_size

            # 计算块的中心索引
            center_row = (row_start + row_end) // 2
            center_col = (col_start + col_end) // 2

            # 将中心索引添加到列表中
            center_indices[n_b, :] = [center_col, center_row]

            n_b += 1
            
        sample_index = np.ravel_multi_index(center_indices.T, (height, width), order='F')

    return sample_index


def divide_image(img, m, n, idx): # 分割成m行n列, 并取出第几行第几列
    
    if img.ndim == 2:
        img = img[np.newaxis, :, :]
        
    # height: m, width: n
    s_max, s_min = max(m, n), min(m, n)
    if m >= n:
        i, j = idx // m, idx % m
        axis_1, axis_2 = 1, 2
    else:
        i, j = idx // n, idx % n
        axis_1, axis_2 = 2, 1
        
    split_1 = np.array_split(img, s_max, axis=axis_1)
    del img
    new_img = np.asarray(split_1[j])
    del split_1
    split_2 = np.asarray(np.array_split(new_img, s_min, axis=axis_2)[i]).squeeze()
    del new_img
    
    # print(split_2.shape)
    
    return split_2

################################################################

if is_train:
    is_split = False
    if is_block:
        block_size = block_size
else:
    is_block = False

prj, geo = None, None

print('file name:')
print('\033[0;31m', file_filter_data, '\033[0m')

""" pick valid pixels based on the roi mask """
ori_width, ori_height, _, _, _, _, boundary_image, _ = read_images(img_directory + '/Area_boundary.dat')
print('height:', boundary_image.shape[0], 'width', boundary_image.shape[1])
if is_split:
    boundary_image = divide_image(boundary_image, m, n, idx)
width, height = boundary_image.shape[1], boundary_image.shape[0]
boundary_reshape = np.transpose(boundary_image, (1, 0)).reshape((width * height)).astype(np.int16) 

if is_block:
    sample_reshape = np.ones((width * height)).astype(np.int16)
    sample_index = spatial_sample(height, width, block_size)
    sample_reshape[sample_index] = 0

if is_block:
    roi_indices = np.where((boundary_reshape == 0) & (sample_reshape == 0))[0]
    num_roi = np.sum((boundary_reshape == 0) & (sample_reshape == 0))
    del sample_reshape
else:
    roi_indices = np.where((boundary_reshape == 0))[0]
    num_roi = np.sum((boundary_reshape == 0))

np.save(output_directory + '/Roi_index', roi_indices)
print('vaild roi pixels: ', num_roi)

# roi_indices = np.load(output_directory + '/Roi_index.npy')
# num_roi = len(roi_indices)
# print('vaild roi pixels: ', num_roi)

""" input ndvi timeseries data """
_, _, n_images, _, _, _, ndvi_image, _ = read_images(img_directory + '/NDVI_timeseries_fine.tif')
if is_split:
    ndvi_image = divide_image(ndvi_image, m, n, idx)

""" new empty arrays """
sr_timeseries = np.zeros((num_roi, sr_bands, n_images)).astype(np.float32)
mask_timeseries = np.zeros((num_roi, 1, n_images)).astype(np.int16)
ndvi_timeseries = np.zeros((num_roi, 1, n_images)).astype(np.float32)

""" process data image by image """
with tqdm(range(n_images), ncols=100) as pbar:

    for i_image in pbar:

        _, _, _, _, prj, geo, data_i, _ = read_images(raw_directory + '/' + file_filter_data[i_image])
        _, _, _, _, _, _, qa_i, _ = read_images(mask_directory + '/' + file_filter_qa[i_image])
        _, _, _, _, _, _, cp_i, _ = read_images(mask_directory + '/' + file_filter_cp[i_image])
        _, _, _, _, _, _, cs_i, _ = read_images(mask_directory + '/' + file_filter_cs[i_image])
        
        if is_dismatch:
            data_i = data_i[:, 0:ori_height, 0:ori_width]
            qa_i = qa_i[:, 0:ori_height, 0:ori_width]
            cp_i = cp_i[0:ori_height, 0:ori_width]
            cs_i = cs_i[:, 0:ori_height, 0:ori_width]
            
        if is_split:
            data_i = divide_image(data_i, m, n, idx)
            print(data_i.shape)
            qa_i = divide_image(qa_i, m, n, idx)
            cp_i = divide_image(cp_i, m, n, idx)
            cs_i = divide_image(cs_i, m, n, idx)
        
        ndvi_i = ndvi_image[i_image, :, :]
        
        sr_i = data_i * scale_factor
        
        """ process cloud and cloud shadow mask """
        qa_60_i = qa_i[0, :, :]
        qa_60_i[np.isnan(qa_60_i)] = 1111
        cloud_qa_60_i = cloud_s2(qa_60_i)

        scl_i = qa_i[1, :, :]
        scl_i[np.isnan(scl_i)] = 9
        conditions_scl = (scl_i == 1) | (scl_i == 2) | (scl_i == 3) | (scl_i == 7) | (scl_i == 8) | (scl_i == 9) | (scl_i == 10)
        cloud_scl_i = np.where(conditions_scl, 1, 0)
        
        cp_i[np.isnan(cp_i)] = 100
        conditions_cp = (cp_i >= 40)
        cloud_cp_i = np.where(conditions_cp, 1, 0)
        
        cs_1_i = cs_i[0, :, :]
        cs_2_i = cs_i[1, :, :]
        cs_1_i[np.isnan(cs_1_i)], cs_2_i[np.isnan(cs_2_i)] = 0, 0
        conditions_cs = (cs_1_i <= 0.75) | (cs_2_i <= 0.75)
        cloud_cs_i = np.where(conditions_cs, 1, 0)
        
        # print(cloud_qa_60_i.shape, cloud_scl_i.shape, cloud_cp_i.shape, cloud_cs_i.shape)
        cloud_all_i = np.any([cloud_qa_60_i, cloud_scl_i, cloud_cp_i, cloud_cs_i], axis=0)
        
        """ remove invalid values """
        # remove invalid ndvi values
        ind_bad_ndvi = np.where((boundary_image == 0) & ((ndvi_i >= 1) | (ndvi_i <= -1) | (ndvi_i == 0) | np.isnan(ndvi_i)))
        ind_r, ind_c = ind_bad_ndvi[0], ind_bad_ndvi[1]

        for i_bad in range(len(ind_r)):
            window_size = 3  # 初始窗口大小
            found_valid_values = False  # 标志是否找到有效值
            while (found_valid_values == False):                            
                cnt_valid = 0, 0  # 初始化用于计算平均值的变量                            
                # 遍历窗口中的元素
                a1, a2 = max(0, ind_r[i_bad] - window_size), min(height, ind_r[i_bad] + window_size + 1)
                b1, b2 = max(0, ind_c[i_bad] - window_size), min(width, ind_c[i_bad] + window_size + 1)
                ndvi_window = ndvi_i[a1 : a2, b1 : b2]
                idx_valid = np.where((~np.isnan(ndvi_window)) & (ndvi_window < 1) & (ndvi_window > -1) & (ndvi_window != 0))
                cnt_valid = np.sum((~np.isnan(ndvi_window)) & (ndvi_window < 1) & (ndvi_window > -1) & (ndvi_window != 0))
                # 如果窗口内有有效值，计算平均值并替代invalid值，并设置 found_valid_values 为 True
                if cnt_valid >= 3:
                    ndvi_i[ind_r[i_bad], ind_c[i_bad]] = np.mean(ndvi_window[idx_valid])
                    found_valid_values = True
                else:
                    window_size += 5  # 否则，扩大窗口范围
                    
                # for r in range(ind_r[i_bad] - window_size, ind_r[i_bad] + window_size + 1):
                #     for c in range(ind_c[i_bad] - window_size, ind_c[i_bad] + window_size + 1):
                #         # 检查索引是否在有效范围内
                #         if (0 <= r < height) and (0 <= c < width) and (np.isnan(ndvi_i[r, c]) == False) and (-1 < ndvi_i[r, c] < 1) and (ndvi_i[r, c] != 0):
                #             total += ndvi_i[r, c]
                #             count += 1
                # # 如果窗口内有有效值，计算平均值并替代invalid值，并设置 found_valid_values 为 True
                # if count >= 3:
                #     ndvi_i[ind_r[i_bad], ind_c[i_bad]] = 1.0 * total / count
                #     found_valid_values = True
                # else:
                #     window_size += 1  # 否则，扩大窗口范围
                
                # print(window_size)
                # 如果窗口范围太大，停止寻找直接设置 found_valid_values 为 True，并设置ndvi为0
                if window_size >= 0.3 * min(height, width):
                    if cnt_valid != 0:
                        ndvi_i[ind_r[i_bad], ind_c[i_bad]] = np.mean(ndvi_window[idx_valid])
                        found_valid_values = True
                    else:
                        ndvi_i[ind_r[i_bad], ind_c[i_bad]] = 0.001
                        found_valid_values = True

        # remove invalid sr values
        for i_band in range(sr_bands):
            index_bad_sr = (boundary_image == 0) & ((sr_i[i_band, :, :] >= 1) | (sr_i[i_band, :, :] <= 0) | np.isnan(sr_i[i_band, :, :]))
            cloud_all_i[index_bad_sr] = 1
            sr_i[i_band, :, :][index_bad_sr] = 0
        
        sr_reshape_i = np.transpose(sr_i, (2, 1, 0)).reshape((width * height, sr_bands)).astype(np.float32)
        sr_roi_i = sr_reshape_i[roi_indices, :]
        sr_timeseries[:, :, i_image] = sr_roi_i
        del sr_i, sr_reshape_i, sr_roi_i
        gc.collect()
        
        ndvi_reshape_i = np.transpose(ndvi_i, (1, 0)).reshape((width * height)).astype(np.float32)
        ndvi_roi_i = ndvi_reshape_i[roi_indices]
        ndvi_timeseries[:, 0, i_image] = ndvi_roi_i
        del ndvi_i, ndvi_reshape_i, ndvi_roi_i
        gc.collect()
        
        cloud_reshape_i = np.transpose(cloud_all_i, (1, 0)).reshape((width * height)).astype(np.int16)
        cloud_roi_i = cloud_reshape_i[roi_indices]
        mask_timeseries[:, 0, i_image] = cloud_roi_i
        del cloud_all_i, cloud_reshape_i, cloud_roi_i
        gc.collect()

del ndvi_image
gc.collect()
np.save(output_directory + '/Cloud_train_roi', mask_timeseries)
lstm_intervals(output_directory, mask_timeseries, dates_file, date_reference, n_images)
del mask_timeseries
gc.collect()
np.save(output_directory + '/NDVI_train_roi', ndvi_timeseries)
del ndvi_timeseries
gc.collect()
np.save(output_directory + '/SRef_train_roi', sr_timeseries)
del sr_timeseries
gc.collect()

t2 = time.perf_counter()
print('程序运行时间: %s s' % (t2 - t1))
