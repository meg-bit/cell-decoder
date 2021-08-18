import numpy as np
from scipy import ndimage
from scipy import stats
from skimage.filters import threshold_yen
import matplotlib.pyplot as plt
    

def mask_select(masks_mem, df_cell_id, metric):
    no_double = df_cell_id.barcode[df_cell_id[metric]==True]
    mask_color_nd = np.zeros((*masks_mem.shape, 3))
    for i in no_double.index:
        if i>0:
            mask_color_nd[np.array(masks_mem)==i] = color_dict[no_double[i]]
    return mask_color_nd
    
def mask_all(masks_mem, cell_id):
    mask = np.zeros(masks_mem.shape)
    for i, l in enumerate(cell_id[1:]):
        mask[masks_mem==i+1] = l+1  # avoid label l being 0 because the background is 0
        
    mask_color = np.zeros((*mask.shape, 3))
    for i in range(len(color_dict)): 
        mask_color[mask==i+1] = color_dict[i]
    return mask_color
    
    
def sample_size(r, alpha, beta):
    C = 0.5 * np.log((1+r)/(1-r+1e-6))  # +1e-6 is to avoid zero division
    z_alpha = np.abs(stats.norm.ppf(alpha/2))
    z_beta = stats.norm.ppf(beta)
    N = int(np.abs(((z_alpha+z_beta)/(C)) ** 2 + 3))  # int(np.abs()) is to convert to an integer
    return N


def minmax(x, min_all, max_all):
    y = (x-min_all)/max_all
    return y


def argmax_thresh(a, axis, thresh):
    rows_too_small = np.where(np.max(a, axis=1) < thresh)
    my_argmax = a.argmax(axis=axis)
    my_argmax[rows_too_small] = -1
    return my_argmax

    
def cal_metric(cell_table, codebook):
    """
    Calculate the correlation between the cell_table and the codebook
    """
    cell_norm = np.sqrt(np.sum(np.power(cell_table, 2), axis=1))
    cell_corr = cell_table.dot(codebook.T) / np.reshape(cell_norm + 1e-6, (-1,1))  # add 1e-6 to avoid the denominator being 0
    return cell_corr


def create_celltable(Xyen, masks_mem, up_adjust, left_adjust):
    """
    Given the membrane segmentation and the registered images with amplicon spots, generate the cell-by-amplicon table.
    
    Args:
        Xthresh: X after thresholding
        masks_mem:
        n_std: an int of the number of standard deviation above the mean for the peak_thresh
            
    Returns:

    """
    cell_table = np.zeros((len(np.unique(masks_mem)), Xyen.shape[0]))

    for k in range(Xyen.shape[0]):  # for the kth image
        # Get local maximum values of desired neighborhood
        max_fil = ndimage.maximum_filter(Xyen[k,], size=(1, 2, 2))
        peak_thresh = max_fil.mean() + max_fil.std() * 4

        # find areas greater than peak_thresh
        labels, num_labels = ndimage.label(max_fil > peak_thresh)

        # Get the positions of the maxima
        coords = ndimage.measurements.center_of_mass(Xyen[k,], 
                                                     labels=labels, 
                                                     index=np.arange(1, num_labels + 1))

        for _, m1, m2 in coords:
            m1 = int(np.round(m1))
            m2 = int(np.round(m2))
            mem_id = masks_mem[m1+up_adjust, m2+left_adjust]  # important to match the coordinates if images are trimmed
            if mem_id>0: # 0 is background
                cell_table[mem_id, k] += 1
    return cell_table


# def lower_thresh(Xthresh):
#     Xyen = np.zeros(Xthresh.shape)
#     for i in range(Xthresh.shape[0]):
#         image = Xthresh[i, 0]
#         thresh = threshold_yen(image)
#         binary = image > thresh
#         Xyen[i, 0, ] = binary
#     return Xyen


def upper_collapse(Xnorm, upper):
    Xthresh = Xnorm.copy()
    for i in range(Xthresh.shape[0]):
        single = Xthresh[i, 0,]
        single[single > upper[i]] = upper[i]
        Xthresh[i, 0] = single
    return Xthresh


def upper_thresh(Xnorm, n_channels, n_cycles, min_bin):
    upper = np.zeros(n_channels * n_cycles)
    n_pixels = np.prod(Xnorm.shape[2:4])  # Xnorm.shape[2, 4] is the dimension of the 2D image
    for i in range(Xnorm.shape[0]):
        hist, edges = np.histogram(Xnorm[i, 0].ravel(), bins=256, range=(0, 1))
        z = np.cumsum(hist)
        dz = np.diff(z)
        idx = np.array(np.where(dz >= min_bin)).ravel()[-1]  # 3 set a param here
        upper[i] = edges[idx]
    return upper


# def upper_thresh(Xnorm, n_channels, n_cycles, min_bin):
#     upper = np.zeros(n_channels * n_cycles)
#     n_pixels = np.prod(Xnorm.shape[2:4])  # Xnorm.shape[2, 4] is the dimension of the 2D image
#     for i in range(Xnorm.shape[0]):
#         hist, edges = np.histogram(Xnorm[i, 0].ravel(), bins=256, range=(0, 1))
#         z = np.cumsum(hist)
#         idx = np.array(np.where(z >= (n_pixels-min_bin))).ravel()[0]
#         upper[i] = edges[idx]
#     return upper


# def limit_upper(Xnorm, upper):
#     Xthresh = Xnorm.copy()
#     for i in range(Xthresh.shape[0]):
#         single = Xthresh[i, 0,]
#         single[single > upper[i]] = upper[i]
#         Xthresh[i, 0] = single
#     return Xthresh


# def subtract_background(X, window_size):


color_dict = {
    0: (0,0,255),
    1: (0,255,0),
    2: (255, 0, 0),
    3: (0,255,255),
    4: (255,0,255),
    5: (255,255,0),
    6: (255,127,0),
    7: (0,127,0),
    8: (127,0,0),
    9: (255,255,255)
}


def remove_border(X, up, down, left, right):
    return X[:, :, up:down, left:right]
