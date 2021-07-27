from scipy import ndimage


# def mean_std(max_fil, num_std):
#     return max_fil.mean() + max_fil.std() * num_std

def create_celltable(Xthresh, masks_mem, n_std, up_adjust, left_adjust):
    """
    Given the membrane segmentation and the registered images with amplicon spots, generate the cell-by-amplicon table.
    
    Args:
        Xthresh: X after thresholding
        masks_mem:
        n_std: an int of the number of standard deviation above the mean for the peak_thresh
            
    Returns:

    """
    cell_table = np.zeros((len(np.unique(masks_mem)), Xthresh.shape[0]))

    for k in range(Xthresh.shape[0]):  # for the kth image
        # Get local maximum values of desired neighborhood (size of the amplicons)
        max_fil = ndimage.maximum_filter(Xthresh[k,], size=(1, 2, 2))

        # Threshold the image to find locations of interest
        # assuming 6 standard deviations above the mean of the image
        peak_thresh = max_fil.mean() + max_fil.std() * n_std

        # find areas greater than peak_thresh
        labels, num_labels = ndimage.label(max_fil > peak_thresh)

        # Get the positions of the maxima
        coords = ndimage.measurements.center_of_mass(Xthresh[k,], 
                                                     labels=labels, 
                                                     index=np.arange(1, num_labels + 1))

        # # Get the maximum value in the labels
        # values = ndimage.measurements.maximum(img, labels=labels, index=np.arange(1, num_labels + 1))
        # # https://stackoverflow.com/questions/55453110/how-to-find-local-maxima-of-3d-array-in-python

        for _, m1, m2 in coords:
            m1 = int(np.round(m1))
            m2 = int(np.round(m2))
            mem_id = masks_mem[m1+up_adjust, m2+left_adjust]  # important to match the coordinates if images are trimmed
            if mem_id>0: # 0 is background
    #             cell_table.loc[mem_id, k] += 1
                cell_table[mem_id, k] += 1
    return cell_table