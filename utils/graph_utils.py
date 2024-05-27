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

def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = torch.concatenate([a, torch.zeros([ori_len, node_number - ori_len], device=a.device)], axis=-1)
    a = torch.concatenate([a, torch.zeros([node_number - ori_len, node_number], device=a.device)], axis=0)
    return a

def mask_adjs(adjs, node_flags):
    """

    :param adjs:  B x N x N or B x C x N x N
    :param node_flags: B x N
    :return:
    """
    # assert node_flags.sum(-1).gt(2-1e-5).all(), f"{node_flags.sum(-1).cpu().numpy()}, {adjs.cpu().numpy()}"
    if len(adjs.shape) == 4:
        node_flags = node_flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * node_flags.unsqueeze(-1)
    adjs = adjs * node_flags.unsqueeze(-2)
    return adjs

def add_self_loop_if_not_exists(adjs):
    if len(adjs.shape) == 4:
        return adjs + torch.eye(adjs.size()[-1], device=adjs.device).unsqueeze(0).unsqueeze(0)
    return adjs + torch.eye(adjs.size()[-1], device=adjs.device).unsqueeze(0)

def check_adjs_symmetry(adjs):
    tr_adjs = adjs.transpose(-1, -2)
    assert (adjs - tr_adjs).abs().sum([0, 1, 2]) < 1e-2