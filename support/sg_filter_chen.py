import numpy as np
from scipy.signal import savgol_filter


def sg_filter_chen(vector_in, hw_1, hw_2, d_1, d_2):

    w_1 = 2 * hw_1 + 1
    w_2 = 2 * hw_2 + 1

    data_shape = vector_in.shape
    nl = data_shape[1]
    nb = data_shape[0]
    first_sg = savgol_filter(vector_in, w_1, d_1, mode='nearest', axis=0)

    delta = abs(vector_in - first_sg)
    delta_max = np.max(delta, axis=0).reshape(1, nl).repeat(repeats=nb, axis=0)
    fl = np.select([vector_in >= first_sg, vector_in < first_sg], [1.0, 1.0 * (1 - delta / delta_max)])
    gdis = np.sum(fl * abs(vector_in - first_sg), axis=0)
    gdis_num = nl
    gdis_addr = range(gdis_num)
    gdis2 = gdis.copy()
    iter_sg_temp = first_sg.copy()
    img_sg = np.max([vector_in, first_sg], axis=0)

    loop_times = 0
    while gdis_num > 0 and loop_times < 1000:
        loop_times += 1
        pre_sg = img_sg[:, gdis_addr]
        fl_temp = fl[:, gdis_addr]
        iterative_sg = savgol_filter(pre_sg, w_2, d_2, mode='nearest', axis=0)
        gdis2[gdis_addr] = np.sum(fl_temp * abs(vector_in[:, gdis_addr] - iterative_sg), axis=0)
        img_sg[:, gdis_addr] = np.max([iterative_sg, pre_sg], axis=0)
        img_sg[:, gdis2 > gdis] = iter_sg_temp[:, gdis2 > gdis]
        iter_sg_temp[:, gdis_addr] = iterative_sg
        gdis_addr = np.where(gdis2 < gdis)
        gdis_num = np.sum(gdis2 < gdis)
        gdis = gdis2.copy()


    return img_sg


def sg_filter_chen_w(vector_in, trend_line, hw_1, hw_2, d_1, d_2, weight):

    w_1 = 2 * hw_1 + 1
    w_2 = 2 * hw_2 + 1
    
    data_shape = vector_in.shape
    nl = data_shape[1]
    nb = data_shape[0]
    first_sg = savgol_filter(trend_line, w_1, d_1, mode='nearest', axis=0)

    delta = abs(vector_in - first_sg)
    delta_max = np.max(delta, axis=0).reshape(1, nl).repeat(repeats=nb, axis=0)
    fl = np.select([vector_in >= first_sg, vector_in < first_sg], [1.0, 1.0 * (1 - delta / delta_max) * weight])
    gdis = np.sum(fl * abs(vector_in - first_sg), axis=0)
    gdis_num = nl
    gdis_addr = range(gdis_num)
    gdis2 = gdis.copy()
    iter_sg_temp = first_sg.copy()
    img_sg = np.max([vector_in, first_sg], axis=0)

    loop_times = 0
    while gdis_num > 0 and loop_times < 1000:
        loop_times += 1
        pre_sg = img_sg[:, gdis_addr]
        fl_temp = fl[:, gdis_addr]
        iterative_sg = savgol_filter(pre_sg, w_2, d_2, mode='nearest', axis=0)
        gdis2[gdis_addr] = np.sum(fl_temp * abs(vector_in[:, gdis_addr] - iterative_sg), axis=0)
        img_sg[:, gdis_addr] = np.max([iterative_sg, pre_sg], axis=0)
        img_sg[:, gdis2 > gdis] = iter_sg_temp[:, gdis2 > gdis]
        iter_sg_temp[:, gdis_addr] = iterative_sg
        gdis_addr = np.where(gdis2 < gdis)
        gdis_num = np.sum(gdis2 < gdis)
        gdis = gdis2.copy()


    return img_sg
