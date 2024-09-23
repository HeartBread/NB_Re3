import os
import time
import sys
from support.utils import mosaic_images

sys.path.append('/data/newhome/hongtao/Pyproject/Reflectance_Reconstruction')
# os.environ['PROJ_LIB'] = r'E:\Hongtao\Python_3.11\Lib\site-packages\osgeo\data\proj'

t1 = time.perf_counter()

file_prefix = 'S2_Area_1_'
data_directory = output_directory =  '/data/newhome/hongtao/Area_1/raw_image'
dates_file = open(data_directory + '/dates_timeseries.txt')
dates_images = dates_file.read().splitlines()
dates_file.close()
# print(dates_images)
# dates_images = ['20220604']

for i_date in dates_images:
    
    print(i_date)
    Image_1 = data_directory + '/' + file_prefix + i_date + '-0000000000-0000000000.tif'
    Image_2 = data_directory + '/' + file_prefix + i_date + '-0000000000-0000007424.tif'
    Image_3 = data_directory + '/' + file_prefix + i_date + '-0000007424-0000000000.tif'
    Image_4 = data_directory + '/' + file_prefix + i_date + '-0000007424-0000007424.tif'
    Image_5 = data_directory + '/' + file_prefix + i_date + '-0000014848-0000000000.tif'
    Image_6 = data_directory + '/' + file_prefix + i_date + '-0000014848-0000007424.tif'
    
    output_1 = output_directory + '/' + file_prefix + '1_' + i_date + '.tif'
    output_2 = output_directory + '/' + file_prefix + '2_' + i_date + '.tif'
    output_3 = output_directory + '/' + file_prefix + '3_' + i_date + '.tif'
    output_4 = output_directory + '/' + file_prefix + '4_' + i_date + '.tif'
    output = output_directory + '/' + file_prefix + i_date + '.tif'
    
    mosaic_images(Image_1, Image_2, output_1, 'height')
    mosaic_images(Image_3, Image_4, output_2, 'height')
    mosaic_images(Image_5, Image_6, output_3, 'height')
    
    mosaic_images(output_1, output_2, output_4, 'width')
    mosaic_images(output_4, output_3, output, 'width')

    
    os.remove(output_1)
    os.remove(output_2)
    os.remove(output_3)
    os.remove(output_4)

print('程序运行时间: %s s' % (t2 - t1))
