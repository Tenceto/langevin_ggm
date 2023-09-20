import numpy as np
import torch


def adj_matrix_to_vec(H):
    h = H[np.triu_indices_from(H, k=1)]
    return h

def vec_to_adj_matrix(h, output_size=None):
    if output_size is None:
        output_size = (1 + np.sqrt(1 + 8 * len(h))) / 2
        output_size = int(output_size)
    H = np.zeros((output_size, output_size))
    H[np.triu_indices_from(H, k=1)] = h
    H = H + H.T
    return H

def random_nan_replacement(x, n_choose, one_zero_ratio=0.5, n_proportional=True):
    x = x.astype(float)

    if n_proportional:
        n_choose = int(n_choose * len(x)) - np.isnan(x).sum()

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

def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = torch.concatenate([a, torch.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = torch.concatenate([a, torch.zeros([node_number - ori_len, node_number])], axis=0)
    return a

def lambda_generic(a, b, c):
	return lambda n: 1 / (a * np.log(n) ** 2 + b * np.log(n) + c)