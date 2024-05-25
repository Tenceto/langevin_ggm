import numpy as np


def random_nan_replacement(x, n_choose, n_proportional=True, one_zero_ratio=None):
    x = x.astype(float)

    if n_proportional:
        n_choose = int(n_choose * len(x)) - np.isnan(x).sum()

    if one_zero_ratio is None:
        one_zero_ratio = x.mean()

    num_ones = np.random.binomial(1, p=one_zero_ratio, size=n_choose).sum()
    num_zeros = n_choose - num_ones

    idxs_0 = np.where(x == 0)[0]
    idxs_1 = np.where(x == 1)[0]

    if len(idxs_0) < num_zeros:
        num_zeros = len(idxs_0)
        num_ones = n_choose - num_zeros
    elif len(idxs_1) < num_ones:
        num_ones = len(idxs_1)
        num_zeros = n_choose - num_ones
    
    mask_0 = np.random.choice(idxs_0, num_zeros, replace=False)
    mask_1 = np.random.choice(idxs_1, num_ones, replace=False)

    x[mask_0] = np.nan
    x[mask_1] = np.nan

    return x

def lambda_generic(n, a, b, c):
	return 1 / (a * np.log(n) ** 2 + b * np.log(n) + c)
