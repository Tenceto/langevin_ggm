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

def _lambda_generic(n, a, b, c):
	return 1 / (a * np.log(n) ** 2 + b * np.log(n) + c)

def lambda_glasso_selector(graph_type, nans, nans_proportional, one_zero_ratio):
    if graph_type == "ergm":
        if nans == 30 and nans_proportional == False and one_zero_ratio == 0.5:
            return lambda n: _lambda_generic(n, 2.69135693, 4.61820941, -13.35200138)
        elif nans == 0.2 and nans_proportional == True and one_zero_ratio is None:
            return lambda n: _lambda_generic(n, 4.61577605, -28.03207726, 78.38257537)
    elif graph_type == "grids":
        if nans == 30 and nans_proportional == False and one_zero_ratio == 0.5:
            return lambda n: _lambda_generic(n, 4.87366705, -19.5206655, 48.02892549)
        elif nans == 0.1 and nans_proportional == True and one_zero_ratio is None:
            return lambda n: _lambda_generic(n, 3.72769248, -18.51843188, 56.53676354)
        # elif nans == 0.25 and nans_proportional == True and one_zero_ratio is None:
        #     return lambda n: _lambda_generic(n, 4.61577605, -28.03207726, 78.38257537)
        elif nans == 0.2 and nans_proportional == True and one_zero_ratio is None:
            return lambda n: _lambda_generic(n, 4.61577605, -28.03207726, 78.38257537)
        elif nans == 0.5 and nans_proportional == True and one_zero_ratio is None:
            return lambda n: _lambda_generic(n, 3.86195781, -21.83851068, 66.09558828)
        elif nans == 1.0 and nans_proportional == True and one_zero_ratio is None:
            return lambda n: _lambda_generic(n, 3.8484164, -22.08741836, 67.14491246)
    elif graph_type == "barabasi":
        if nans == 30 and nans_proportional == False and one_zero_ratio == 0.5:
            return lambda n: _lambda_generic(n, 3.93966846, -6.46266204, 10.46412147)
        elif nans == 0.1 and nans_proportional == True and one_zero_ratio is None:
            return lambda n: _lambda_generic(n, 5.64944576, -40.62433766, 113.83848422)
        elif nans == 0.1 and nans_proportional == True and one_zero_ratio == 0.2:
            return lambda n: _lambda_generic(n, 15.3884894, -138.48359488, 363.68821508)
    elif graph_type == "deezer":
        if nans == 0.05 and nans_proportional == True and one_zero_ratio == 0.5:
            return lambda n: _lambda_generic(n, 0.09569316, -21.02993316, 37.3193769)
        elif nans == 0.5 and nans_proportional == True and one_zero_ratio is None:
            return lambda n: _lambda_generic(n, 2.10615704, -5.49361124, 18.04472622)
        elif nans == 1.0 and nans_proportional == True and one_zero_ratio is None:
            return lambda n: _lambda_generic(n, 3.16160777, -17.52335464, 45.6692579)
    else:
        return None