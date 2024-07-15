import numpy as np
from scipy import ndimage
from scipy import stats
from skimage import filters
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

def remove_borders(X, up, down, left, right):
    """
    Crop the input array X by removing the specified border.

    Args:
        X (numpy.ndarray): Input array to be cropped.
        up (int): Number of rows to remove from the top.
        down (int): Number of rows to remove from the bottom.
        left (int): Number of columns to remove from the left.
        right (int): Number of columns to remove from the right.

    Returns:
        numpy.ndarray: Cropped array.
    """
    return X[:, :, up:down, left:right]

def substract_background(Xcenter, sigma):
    """
    Subtract a Gaussian-smoothed version of Xcenter from itself to remove background noise.

    Args:
        Xcenter (numpy.ndarray): Input array representing the center data.
        sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
        numpy.ndarray: Background-subtracted array with non-negative values.
    """
    Xsmooth = filters.gaussian(Xcenter, sigma=sigma, preserve_range=True)
    Xnorm = Xcenter - Xsmooth
    return Xnorm

def normalize(Xnorm):
    """
    Normalize the input array by scaling its values to the range [0, 1] to make different images in the same range.

    Args:
        Xnorm (numpy.ndarray): Input array to be normalized.

    Returns:
        numpy.ndarray: Normalized array with values in the range [0, 1].
    """
    return (Xnorm - Xnorm.min()) / Xnorm.max()

def calc_upper_threshold(Xnorm, n_channels, n_cycles, min_bin=3):
    """
    Calculate the upper threshold for each channel and cycle.

    Args:
        Xnorm (np.ndarray): Normalized 3D image data.
        n_channels (int): Number of channels.
        n_cycles (int): Number of cycles.
        min_bin (float): Minimum bin difference for threshold calculation.

    Returns:
        np.ndarray: Array of upper thresholds for each channel and cycle.
    """
    upper = np.zeros(n_channels * n_cycles)
    for i in range(Xnorm.shape[0]):
        hist, edges = np.histogram(Xnorm[i, 0].ravel(), bins=256, range=(0, 1))
        z = np.cumsum(hist)
        dz = np.diff(z)
        idx = np.array(np.where(dz >= min_bin)).ravel()[-1]
        upper[i] = edges[idx]
    return upper

def collapse(Xnorm, upper):
     """
    Apply upper threshold collapse to the normalized image data.

    Args:
        Xnorm (np.ndarray): Normalized 3D image data.
        upper (np.ndarray): Array of upper thresholds for each channel and cycle.

    Returns:
        np.ndarray: Collapsed image data after applying upper threshold.
    """
    Xthresh = Xnorm.copy()
    for i in range(Xthresh.shape[0]):
        single = Xthresh[i, 0]
        single[single > upper[i]] = upper[i]
        Xthresh[i, 0] = single
    return Xthresh

def create_celltable(Xthresh, masks_mem, up, left, percentile):
    """
    Create a cell table based on thresholded image data.

    Args:
        Xthresh (np.ndarray): Thresholded 3D image data.
        masks_mem (np.ndarray): Membrane masks (integer labels for each pixel).
        up (int): Vertical offset for matching coordinates.
        left (int): Horizontal offset for matching coordinates.
        percentile (float): Percentile value for thresholding.

    Returns:
        np.ndarray: Cell table with counts for each membrane ID and the number of signal peaks.
    """
    cell_table = np.zeros((np.max(masks_mem), Xthresh.shape[0]))
    for k in range(Xthresh.shape[0]):  # for the kth image
        max_fil = ndimage.maximum_filter(Xthresh[k, 0], size=(3, 3), mode='nearest')
        h = np.percentile(Xthresh[k, 0].ravel(), percentile)
        coords = np.argwhere((max_fil == Xthresh[k, 0]) & (max_fil >= h))
        for m1, m2 in coords:
            m1 = int(np.round(m1))
            m2 = int(np.round(m2))
            mem_id = masks_mem[m1+up, m2+left]  # important to match the coordinates if images are trimmed
            if mem_id>0:
                cell_table[mem_id, k] += 1
    return cell_table

def calc_metric(cell_table, codebook):
    """
    Calculate the cross correlation between the cell_table and the codebook.

    Args:
        cell_table (np.ndarray): Cell table with counts (shape: [n_membranes, n_samples]).
        codebook (np.ndarray): Reference codebook (shape: [n_membranes, n_features]).

    Returns:
        np.ndarray: Correlation matrix between cell_table and codebook.
    """
    cell_norm = np.sqrt(np.sum(np.power(cell_table, 2), axis=1))
    cell_corr = cell_table.dot(codebook.T) / np.reshape(cell_norm + 1e-6, (-1,1))  # add 1e-6 to avoid the denominator being 0
    return cell_corr

def argmax_thresh(a, threshold):
    """
    Finds the index of the maximum value in each row of a 2D array, considering only values above a given threshold.

    Args:
        a (np.ndarray): Input 2D array.
        threshold (float): Threshold value for filtering.

    Returns:
        np.ndarray: Array of indices corresponding to the maximum values in each row. If the maximum value in a row
                    is below the threshold, the corresponding index is set to -1.
    """
    rows_too_small = np.where(np.max(a, axis=1) < threshold)
    my_argmax = a.argmax(axis=1)
    my_argmax[rows_too_small] = -1
    return my_argmax

def create_color_mask(masks_mem, cell_id):
    """
    Create a color mask based on membrane labels.

    Args:
        masks_mem (np.ndarray): Membrane masks (integer labels for each pixel).
        cell_id (np.ndarray): Array of cell IDs (excluding background).

    Returns:
        np.ndarray: Color mask with RGB values.
    """
    color_dict = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (0, 255, 255),
        4: (255, 0, 255),
        5: (255, 255, 0),
        6: (255, 127, 0),
        7: (0, 127, 0),
        8: (127, 0, 0),
        9: (255, 255, 255)
    }
    mask = np.zeros(masks_mem.shape)
    for i, l in enumerate(cell_id[1:]):
        mask[masks_mem == i + 1] = l + 1  # avoid label l being 0 because the background is 0

    mask_color = np.zeros((*mask.shape, 3))
    for i in range(len(color_dict)):
        mask_color[mask == i + 1] = color_dict[i]
    return mask_color
    

    
    








    

























