# stack and crop images into timeseries with fixed size
import time
import sys
import numpy as np
from tqdm import tqdm
import gc
from support.utils import filelist_filter, read_images, arr2raster

sys.path.append('/data/newhome/hongtao/Pyproject/Reflectance_Reconstruction')

t1 = time.perf_counter()

# ******************************************************************************************************
study_area = 'Naidong'
data_directory = '/data/newhome/hongtao' + '/Naidong_Shannan'
img_directory = data_directory + '/raw_image'
mask_directory = data_directory + '/image_mask'
output_directory = data_directory + '/image_data'
file_filter_data = filelist_filter(img_directory, prefix_label='S2_' + study_area + '_', suffix_label = '.dat')
file_filter_qa = filelist_filter(mask_directory, prefix_label='S2_' + study_area + '_qa_', suffix_label = '.tif')
file_filter_cp = filelist_filter(mask_directory, prefix_label='S2_' + study_area + '_Cloud_probability_', suffix_label = '.tif')
file_filter_cs = filelist_filter(mask_directory, prefix_label='S2_' + study_area + '_Cloud_score_', suffix_label = '.tif')

dates_file = open(data_directory + '/dates_timeseries.txt')

n_images = 49
sr_bands = 10
width = 4969
height = 9330
scale_factor = 0.0001
band_name_data = ['2_blue', '3_green', '4_red', '5_red_edge_1', '6_red_edge_2', '7_red_edge_3', '8_nir', '8A_red_edge_4', '11_swir_1', '12_swir_2']
# ******************************************************************************************************


# input: pixel_qa
# output: cloud (true), clear (false)
def cloud_s2(data):

    data = data.astype(np.int16)

    return np.bitwise_and(np.right_shift(data, 10), 11) != 0


# sr_timeseries = []
# mask_timeseries = []

sr_timeseries = np.zeros((n_images * sr_bands, height, width)).astype(np.float32)
mask_timeseries = np.zeros((n_images, height, width)).astype(np.int16)

band_num = None
proj, geo = None, None

print('file name:')
print('\033[0;31m', file_filter_data, '\033[0m')

with tqdm(range(n_images), ncols=100) as pbar:
    
    for i_image in pbar:

        data_i_info = read_images(img_directory + '/' + file_filter_data[i_image])
        qa_i_info = read_images(mask_directory + '/' + file_filter_qa[i_image])
        cp_i_info = read_images(mask_directory + '/' + file_filter_cp[i_image])
        cs_i_info = read_images(mask_directory + '/' + file_filter_cs[i_image])
        
        data_i = data_i_info[6]
        qa_i = qa_i_info[6]
        cp_i = cp_i_info[6]
        cs_i = cs_i_info[6]
        band_num = data_i_info[2]

        sr_i = data_i[0: band_num, :, :] * scale_factor
        
        qa_60_i = qa_i[0, :, :]
        qa_60_i[np.isnan(qa_60_i)] = 0
        cloud_qa_60_i = cloud_s2(qa_60_i)

        scl_i = qa_i[1, :, :]
        scl_i[np.isnan(scl_i)] = 0
        conditions_scl = (scl_i == 1) | (scl_i == 2) | (scl_i == 3) | (scl_i == 7) | (scl_i == 8) | (scl_i == 9) | (scl_i == 10)
        cloud_scl_i = np.where(conditions_scl, 1, 0)
        
        cp_i[np.isnan(cp_i)] = 0
        conditions_cp = (cp_i >= 50)
        cloud_cp_i = np.where(conditions_cp, 1, 0)
        
        cs_1_i = cs_i[0, :, :]
        cs_2_i = cs_i[1, :, :]
        conditions_cs = (cs_1_i <= 0.6) | (cs_2_i <= 0.6)
        cs_1_i[np.isnan(cs_1_i)], cs_2_i[np.isnan(cs_2_i)] = 0, 0
        cloud_cs_i = np.where(conditions_cs, 1, 0)
        
        # print(cloud_qa_60_i.shape, cloud_scl_i.shape, cloud_cp_i.shape, cloud_cs_i.shape)
        cloud_all_i = np.any([cloud_qa_60_i, cloud_scl_i, cloud_cp_i, cloud_cs_i], axis=0)
        cloud_all_i = cloud_all_i[np.newaxis, :, :]
        
        # 去掉nan值, 并设置对应的mask为缺失
        for i_band in range(band_num):
            
            index_bad = ((sr_i[i_band, :, :] >= 1) | (sr_i[i_band, :, :] <= 0) | np.isnan(sr_i[i_band, :, :]))
            cloud_all_i[0, :, :][index_bad] = 1
            sr_i[i_band, :, :][index_bad] = 0
        
        # sr_timeseries.append(sr_i)
        # mask_timeseries.append(cloud_all_i)
        
        sr_timeseries[i_image * sr_bands: i_image * sr_bands + sr_bands, :, :] = sr_i[:, :, :]
        mask_timeseries[i_image, :, :] = cloud_all_i[0, :, :]
        
        del sr_i, cloud_all_i

        if i_image == 0:
            proj = data_i_info[4]
            geo = data_i_info[5]

dates_images = dates_file.read().splitlines()

dates_images = [f'{i+1}_{image}' for i, image in enumerate(dates_images)]
dates_images_repeat_data = []
for i, image in enumerate(dates_images):
    for j in range(band_num):
        new_image_data = f'{image}_({band_name_data[j]})'
        dates_images_repeat_data.append(new_image_data)

dates_file.close()

# mask_timeseries = np.concatenate(mask_timeseries, axis=0)
arr2raster(mask_timeseries, output_directory + '/Mask_timeseries_fine.dat', band_names=dates_images, prj=proj, trans=geo)
del mask_timeseries

# sr_timeseries = np.concatenate(sr_timeseries, axis=0)
arr2raster(sr_timeseries, output_directory + '/SRef_timeseries_fine.dat', band_names=dates_images_repeat_data, prj=proj, trans=geo)

t2 = time.perf_counter()
print('程序运行时间: %s s' % (t2 - t1))
