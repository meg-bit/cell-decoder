import matplotlib.pyplot as plt

# def check_shape(X):
#     return X.shape

# def view_img(X):
#     for i in range(X.shape[0]):
#         plt.figure(dpi=250)
#         plt.imshow(X[i, 0,])
    
def remove_border(X, up, down, left, right):
    return X[:, :, up:down, left:right]

# def subtract_background(X, window_size):


def limit_upper(Xnorm, upper):
    Xthresh = Xnorm.copy()
    for i in range(Xthresh.shape[0]):
        single = Xthresh[i, 0,]
        single[single > upper[i]] = upper[i]
        Xthresh[i, 0] = single
    return Xthresh


