# composite unequally time-intervals timeseries into equally time-intervals timeseries
# (scale of original time series contains scale of synthetic time series)


import time
import numpy as np
from tqdm import trange
from support.utils import read_images, interval_count, arr2raster
import gc
import os
# os.environ['PROJ_LIB'] = r'E:\Hongtao\Python_3.11\Lib\site-packages\osgeo\data\proj'

t1 = time.perf_counter()

output_directory = ndvi_directory = '/data/newhome/hongtao/Area_1/image_data'
# ******************************************************************************************************
width, height, _, _, proj, trans, ndvi_image, _ = read_images(ndvi_directory + '/Area_1_GFSG_2022.tif')
origin_dates = open(ndvi_directory + '/coarse_timeseries.txt')
composite_dates = open(ndvi_directory + '/fine_timeseries.txt')
# ******************************************************************************************************

composite_timeseries = composite_dates.read().splitlines()
origin_timeseries = origin_dates.read().splitlines()
composite_dates.close()
origin_dates.close()

composite_ndvi = np.zeros((len(composite_timeseries), height, width)).astype(np.float32)

for i_date in range(len(composite_timeseries)):
    if composite_timeseries[i_date] in origin_timeseries:
        idx_o = np.where(np.in1d(origin_timeseries, composite_timeseries[i_date]))[0][0]
        composite_ndvi[i_date, :, :] = ndvi_image[idx_o, :, :]
        print(composite_timeseries[i_date], 'true')
    else:
        idx_b = np.searchsorted(origin_timeseries, composite_timeseries[i_date])
        idx_f = idx_b - 1

        d_f = interval_count(origin_timeseries[idx_f], composite_timeseries[i_date])
        d_b = interval_count(origin_timeseries[idx_b], composite_timeseries[i_date])

        w_f = d_b * 1.0 / (d_f + d_b)
        w_b = d_f * 1.0 / (d_f + d_b)

        composite_ndvi[i_date, :, :] = w_f * ndvi_image[idx_f, :, :] + w_b * ndvi_image[idx_b, :, :]
        print(composite_timeseries[i_date], 'false', origin_timeseries[idx_f], origin_timeseries[idx_b])

del ndvi_image
gc.collect()

arr2raster(composite_ndvi / 10000.0, output_directory + '/NDVI_timeseries_fine.tif', prj=proj, trans=trans)

t2 = time.perf_counter()
print('\r 程序运行时间: %s min' % ((t2 - t1) / 60))
