# function support of coding
import os
import time
import numpy as np
from osgeo import gdal
import scipy.signal as ss
import random
import noise


# 读取栅格影像数据, 并返回相应的属性信息
def read_images(data_path):
    dataset = gdal.Open(data_path)  # 打开文件
    img_width = dataset.RasterXSize  # 栅格矩阵的列数
    img_height = dataset.RasterYSize  # 栅格矩阵的行数
    img_bands = dataset.RasterCount  # 栅格影像的波段数

    img_bands_names = []

    for i_band in range(img_bands):
        band_i = dataset.GetRasterBand(i_band + 1)
        band_name = band_i.GetDescription()  # 栅格影像的波段名
        img_bands_names.append(band_name)

    img_prj = dataset.GetProjection()  # 栅格影像的投影信息
    img_geo = dataset.GetGeoTransform()  # 栅格影像的地理坐标, (左上角横坐标, 水平空间分辨率, 旋转参数, 左上角纵坐标, 旋转参数, 垂直空间分辨率)
    img_data = dataset.ReadAsArray(0, 0, img_width, img_height)  # 读取栅格数据集, format: [bands, height, width]
    img_datatype = img_data.dtype

    img_info = [img_width, img_height, img_bands, img_bands_names, img_prj, img_geo, img_data, img_datatype]

    del dataset

    return img_info


# 计算两个时间点之间的时间距离
def interval_count(day_1, day_2):
    time_array1 = time.strptime(day_1, '%Y-%m-%d')
    timestamp_day1 = int(time.mktime(time_array1))
    time_array2 = time.strptime(day_2, '%Y-%m-%d')
    timestamp_day2 = int(time.mktime(time_array2))
    time_interval = abs(timestamp_day1 - timestamp_day2) // 60 // 60 // 24

    return time_interval


# 从文件夹中选取所有固定前缀, 后缀的文件
def filelist_filter(data_directory, prefix_label=None, suffix_label=None):
    file_filter = []
    filelist = [x for x in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, x))]

    for file in filelist:

        prefix = os.path.splitext(file)[0]
        suffix = os.path.splitext(file)[1]

        match (prefix_label, suffix_label):

            case (None, None):
                file_filter.append(file)
            case (prefix_label, None):
                if prefix_label in prefix:
                    file_filter.append(file)
            case (None, suffix_label):
                if suffix == suffix_label:
                    file_filter.append(file)
            case (prefix_label, suffix_label):
                if prefix_label in prefix and suffix == suffix_label:
                    file_filter.append(file)

    file_filter.sort(key=lambda x: int(x.split(prefix_label)[1].split(suffix_label)[0]))  # 将输出列表按照文件名中除去前缀和后缀的剩余部分的顺序排序

    return file_filter


# 将数组输出为固定格式的影像数据
def arr2raster(img_arr, output_path, band_names=None, prj=None, trans=None):
    img_type = None
    data_type = None
    output_suffix = os.path.splitext(output_path)[1]
    output_datatype = img_arr.dtype

    match output_suffix:
        case ('' | '.dat' | '.img'):
            img_type = 'ENVI'  # ENVI .hdr Labelled Raster
        case ('.tif' | '.TIFF'):
            img_type = 'GTiff'  # GeoTIFF File Format

    list1 = ['byte', 'uint8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64', 'cint16', 'cint32', 'cfloat32', 'cfloat64']
    list2 = [gdal.GDT_Byte, gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32, gdal.GDT_Float32, gdal.GDT_Float64, gdal.GDT_CInt16, gdal.GDT_CInt32, gdal.GDT_CFloat32, gdal.GDT_CFloat64]

    if output_datatype in list1:
        data_type = list2[list1.index(output_datatype)]
    else:
        print('数据类型错误！')
        quit()

    driver = gdal.GetDriverByName(img_type)

    if img_type == 'ENVI':
        dst_ds = driver.Create(output_path, img_arr.shape[2], img_arr.shape[1], img_arr.shape[0], data_type, options=['INTERLEAVE=BSQ'])
    if img_type == 'GTiff':
        dst_ds = driver.Create(output_path, img_arr.shape[2], img_arr.shape[1], img_arr.shape[0], data_type, options=['INTERLEAVE=BAND', 'COMPRESS=LZW', 'BIGTIFF=YES'])

    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)

    try:
        for ib in range(img_arr.shape[0]):
            band_i = dst_ds.GetRasterBand(ib + 1)
            band_i.WriteArray(img_arr[ib])
            if band_names is not None:
                band_i.SetDescription(band_names[ib])
    except BaseException as e:
        print('数据写入错误！' + str(e))

    dst_ds.FlushCache()

    del dst_ds


# 计算两个矩阵之间的相关系数, format of x, y: [features, samples], samples of x and y can be unequal, format of out_put: [samples, features]
def matrix_r(x, y, style):
    m_r = 0.0
    old_shape = []

    # 如果x, y都为1维数组, 则新添一个维度
    if x.ndim == 1:
        x = x[:, np.newaxis]

    if y.ndim == 1:
        y = y[:, np.newaxis]

    # 如果x, y都为2维数组, 则强制令其特征维度相同, 否则报错
    if x.ndim == 2 and y.ndim == 2:
        assert x.shape[0] == y.shape[0], 'Features of two inputs are unmatched !'

    # 判断y是否为3维, 3维数组前两个维度代表空间位置信息
    if y.ndim == 3:
        old_shape = (y.shape[0], y.shape[1])
        y = y.reshape(y.shape[0] * y.shape[1], y.shape[2])

        assert x.shape[0] == y.shape[0], 'Features of two inputs are unmatched !'

    if style == 's':  # single to single match, need samples of x and y be unequal
        n = x.shape[0] * x.shape[1]
        sum_xy = np.sum(np.sum(x * y))
        sum_x = np.sum(np.sum(x))
        sum_y = np.sum(np.sum(y))
        sum_x2 = np.sum(np.sum(x * x))
        sum_y2 = np.sum(np.sum(y * y))
        m_r = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))

    if style == 'm':  # single/multi to multi cross-match
        mean_x = np.mean(x, axis=0)  # the mean value of each column
        idx = np.repeat(np.arange(x.shape[1])[np.newaxis, :], x.shape[0], axis=0)
        Var_Xi = x - mean_x[idx]
        SSX = np.sum(abs(Var_Xi) ** 2, axis=0)[np.newaxis, :]  # Sum of squares of columns
        mean_y = np.mean(y, axis=0)  # the mean value of each column
        idy = np.repeat(np.arange(y.shape[1])[np.newaxis, :], y.shape[0], axis=0)
        Var_Yi = y - mean_y[idy]
        SSY = np.sum(abs(Var_Yi) ** 2, axis=0)[np.newaxis, :]  # Sum of squares of columns
        SS = np.dot(SSX.T, SSY)
        m_r = np.dot(np.transpose(Var_Xi, (1, 0)), Var_Yi) / np.sqrt(SS)

    if style == 'cm':  # corresponding multi to multi cross-match, need samples of x and y be equal
        mean_x = np.mean(x, axis=0)  # the mean value of each column
        idx = np.repeat(np.arange(x.shape[1])[np.newaxis, :], x.shape[0], axis=0)
        Var_Xi = x - mean_x[idx]
        SSX = np.sum(abs(Var_Xi) ** 2, axis=0)[np.newaxis, :]  # Sum of squares of columns
        mean_y = np.mean(y, axis=0)  # the mean value of each column
        idy = np.repeat(np.arange(y.shape[1])[np.newaxis, :], y.shape[0], axis=0)
        Var_Yi = y - mean_y[idy]
        SSY = np.sum(abs(Var_Yi) ** 2, axis=0)[np.newaxis, :]  # Sum of squares of columns
        SS = np.dot(SSX.T, SSY)
        m_r = np.dot(np.transpose(Var_Xi, (1, 0)), Var_Yi) / np.sqrt(SS)
        m_r = np.diagonal(m_r)

    if y.ndim == 3:
        m_r = m_r.reshape(old_shape)

    return m_r


# 计算边缘评价指标
def robert_edge(data):

    dataEdg = np.array(data-data)
    kernel1 = np.array([[1, 0], [0, -1]])
    kernel2 = np.array([[0, 1], [-1, 0]])
    robertEdg = np.abs(ss.convolve(data, kernel1, mode='valid')) + np.abs(ss.convolve(data, kernel2, mode='valid'))

    dataEdg[1:-1, 1:-1] = robertEdg[1:, 1:]
    return dataEdg


# 计算纹理评价指标
def lbp_texture(data, tolerance):

    dis = data - data
    dis_temp = np.array([dis[1:-1, 1:-1]] * 8)

    for i in range(0, 8):
        if i == 0:
            kernel = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        elif i == 1:
            kernel = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif i == 2:
            kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        elif i == 3:
            kernel = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        elif i == 4:
            kernel = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        elif i == 5:
            kernel = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        elif i == 6:
            kernel = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
        else:
            kernel = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])

        dis_temp[i, :, :] = ss.convolve(data, kernel, mode='valid')

        gtCenterIdx = np.round(dis_temp[i, ::], 8) > np.round((data[1:-1, 1:-1] + tolerance), 8)
        leCenterIdx = np.round(dis_temp[i, ::], 8) <= np.round((data[1:-1, 1:-1] + tolerance), 8)

        dis_temp[i, gtCenterIdx] = 1
        dis_temp[i, leCenterIdx] = 0

    dis_temp2 = dis_temp[0] * 16 + dis_temp[1] * 8 + dis_temp[2] * 4 + dis_temp[3] * 2 + dis_temp[4] * 1 + dis_temp[5] * 128 + dis_temp[6] * 64 + dis_temp[7] * 32
    dis[1:-1, 1:-1] = dis_temp2

    return dis


# 生成柏林噪声
def perlin_fbm(width, height, octaves=10, persistence=0.5, lacunarity=2.0, seed=None):

    world = np.zeros((width, height))

    unit_random = random.choice(np.arange(0.5, 1, 0.1))
    scale = max(width, height) / unit_random  # 根据地图大小调整尺度

    if seed is None:
        seed = np.random.randint(500)

    for i in range(width):
        for j in range(height):
            value = noise.pnoise2(i / scale, j / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity, base=seed)
            world[i][j] = value

    return world


# 拼接两幅影像, 至少需要有一条边长度一样
def mosaic_images(input_path1, input_path2, output_path, mosaic_boundary):

    # 打开两幅影像
    width1, height1, bands1, _, prj1, geo1, img1, datatype1 = read_images(input_path1)
    width2, height2, bands2, _, prj2, geo2, img2, datatype2 = read_images(input_path2)
    
    assert prj1 == prj2, 'Projections are inconsistent !'
    
    if (datatype1 in ['float32', 'float64']) and (datatype2 in ['float32', 'float64']):
        datatype = np.float32
    elif (datatype1 in ['byte', 'uint8', 'uint16', 'int16', 'uint32', 'int32']) and (datatype2 in ['byte', 'uint8', 'uint16', 'int16', 'uint32', 'int32']):
        datatype = np.int16
    else:
        print('Data types are inconsistent ! Datatype_1 = ', datatype1, ', Datatype_2 = ', datatype2)
        exit()

    # 确保两幅影像的高度一致
    if mosaic_boundary == 'width':
        
        assert width1 == width2
        
        print('width matched')
        new_width = width1
        new_height = height1 + height2
        assert bands1 == bands2, 'Number of bands are unmatched !'
        new_image = np.zeros((bands1, new_height, new_width)).astype(datatype)
        new_image[:, 0: height1, :] = img1
        new_image[:, height1:, :] = img2

        new_geo = list(geo1)
        new_geo[1] = geo1[1]  # 保持原有的像素大小
        new_geo[0] = min(geo1[0], geo2[0])  # 选择左上角较小的经度

        arr2raster(new_image, output_path, prj=prj1, trans=new_geo)

    elif mosaic_boundary == 'height':
        
        assert height1 == height2

        print('height matched')
        new_width = width1 + width2
        new_height = height1
        assert bands1 == bands2, 'Number of bands are unmatched !'
        new_image = np.zeros((bands1, new_height, new_width)).astype(datatype)
        new_image[:, :, 0: width1] = img1
        new_image[:, :, width1:] = img2

        new_geo = list(geo1)
        new_geo[1] = geo1[1]  # 保持原有的像素大小
        new_geo[0] = min(geo1[0], geo2[0])  # 选择左上角较小的经度

        arr2raster(new_image, output_path, prj=prj1, trans=new_geo)
    
    else:
        
        print('Please check the size of images, width and height are unmatched !')


# 计算归一化型指数 (b1 - b2)/ (b1 + b2)
def normalized_index(b1, b2):

    upper = b1 - b2
    below = b1 + b2

    return np.divide(upper, below)
