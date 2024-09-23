import numpy as np
import math
from tqdm import tqdm


def bicubic(coarse_img, ratio, alpha):
    
    (n_coarse, coarse_h, coarse_w) = coarse_img.shape
    coarse_img = coarse_img.astype(np.float32)
    fine_h = np.fix(coarse_h * ratio + 1).astype(np.int16)
    fine_w = np.fix(coarse_w * ratio + 1).astype(np.int16)
    fine_img = np.zeros((n_coarse, fine_h, fine_w))

    top, bottom, left, right, corner = get_padding(coarse_img, coarse_h, coarse_w, n_coarse)
    
    with tqdm(range(fine_h), ncols=100) as pbar:
        for i in pbar:
            for j in range(fine_w):
                x = i / ratio
                y = j / ratio
                x_int = int(math.floor(x))
                y_int = int(math.floor(y))

                for x_diff in range(-1, 3):
                    for y_diff in range(-1, 3):
                        neib_x = x_int + x_diff
                        neib_y = y_int + y_diff

                        if neib_x < 0:
                            if neib_y < 0:
                                fine_img[:, i, j] += corner[0, :] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)
                            elif neib_y >= coarse_w:
                                fine_img[:, i, j] += corner[2, :] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)
                            else:
                                fine_img[:, i, j] += top[:, neib_y] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)
                        elif neib_x >= coarse_h:
                            if neib_y < 0:
                                fine_img[:, i, j] += corner[1, :] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)
                            elif neib_y >= coarse_w:
                                fine_img[:, i, j] += corner[3, :] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)
                            else:
                                fine_img[:, i, j] += bottom[:, neib_y] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)
                        elif neib_y < 0:
                            fine_img[:, i, j] += left[:, neib_x] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)
                        elif neib_y >= coarse_w:
                            fine_img[:, i, j] += right[:, neib_x] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)
                        else:
                            fine_img[:, i, j] += coarse_img[:, neib_x, neib_y] * get_weight(x - neib_x, alpha) * get_weight(y - neib_y, alpha)

    return fine_img


def get_weight(distance, alpha):
    dis = abs(distance)
    if 0 <= dis < 1:
        return (alpha + 2) * pow(dis, 3) - (alpha + 3) * pow(dis, 2) + 1
    elif 1 <= dis <= 2:
        return alpha * pow(dis, 3) - 5 * alpha * pow(dis, 2) + 8 * alpha * dis - 4 * alpha
    else:
        return 0


def get_padding(coarse_img, coarse_h, coarse_w, n_coarse):

    top = coarse_img[:, 0, :]  # [n_coarse, coarse_w]
    bottom = coarse_img[:, coarse_h - 1, :]  # [n_coarse, coarse_w]
    left = coarse_img[:, :, 0]  # [n_coarse, coarse_hh]
    right = coarse_img[:, :, coarse_w - 1]  # [n_coarse, coarse_h]
    
    corner = np.zeros((4, n_coarse))
    # top_left, down_left, top_right, down_right
    corner[0, :] = left[:, 0]  # top_left
    corner[1, :] = left[ :, coarse_h - 1]  # down_left
    corner[2, :] = right[:, 0]  # top_right
    corner[3, :] = right[:, coarse_h - 1]  # down_right
    
    return top, bottom, left, right, corner