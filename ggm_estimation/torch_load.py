import torch
from easydict import EasyDict as edict
from scorematching_gnn.utils.loading_utils import get_score_model
import ggm_estimation.utils as mtx


def load_model(model_file):
    ckp = torch.load(model_file)
    model_config = edict(ckp['config'])
    model = get_score_model(config=model_config, dev="cuda")
    model.load_state_dict(ckp['model'], strict=False)
    model.to("cuda")
    model.eval()

    return model

def score_edp_wrapper(model, nodes, num_sigmas, max_nodes):
	node_flag_init = torch.tensor([1, 0], device="cuda")
	node_flags = node_flag_init.repeat_interleave(torch.tensor([nodes, max_nodes - nodes], device="cuda"))
	node_flags = node_flags.repeat(num_sigmas, 1).float()
	x = torch.zeros((num_sigmas, max_nodes, 1), device="cuda").float()

	def score_fun(A_tilde, U_idxs_triu, sigma_idx):
		model_input = torch.zeros((num_sigmas, max_nodes, max_nodes), device="cuda").float()
		padded_A_tilde = mtx.pad_adjs(A_tilde, max_nodes)
		model_input[num_sigmas - sigma_idx - 1, :, :] = torch.tensor(padded_A_tilde, device="cuda").float()

		with torch.no_grad():
			all_score_levels = model(x, model_input, node_flags)
			vectorized_selected_score = all_score_levels[num_sigmas - sigma_idx - 1, U_idxs_triu[0], U_idxs_triu[1]]
		return vectorized_selected_score.detach()
	
	return score_fun
