import torch
from easydict import EasyDict as edict
from model.gin import GIN
from model.edp_gnn import EdgeDensePredictionGraphScoreNetwork
from utils.graph_utils import pad_adjs


MAX_DEG_FEATURES = 1

NAME_TO_CLASS = {
    'gin': GIN,
}


def get_score_model(config, dev=None, **kwargs):
	if dev is None:
		dev = config.dev
	model_config = list(config.model.models.values())[0]
	in_features = config.dataset.in_feature
	feature_nums = [in_features + MAX_DEG_FEATURES] + model_config.feature_nums
	params = dict(model_config)
	params['feature_nums'] = feature_nums
	params.update(kwargs)

	assert config.model.name == 'edp-gnn'

	def gnn_model_func(**gnn_params):
		merged_params = params
		merged_params.update(gnn_params)
		return NAME_TO_CLASS[model_config.name](**merged_params).to(dev)

	feature_nums[0] = in_features
	score_model = EdgeDensePredictionGraphScoreNetwork(feature_num_list=feature_nums,
														channel_num_list=model_config.channel_num_list,
														max_node_number=config.dataset.max_node_num,
														gnn_hidden_num_list=model_config.gnn_hidden_num_list,
														gnn_module_func=gnn_model_func, dev=dev,
														num_classes=len(config.train.sigmas)).to(dev)
	# logging.info('model: ' + str(score_model))
	return score_model


def load_edpgnn_from_ckpt(model_file, device="cuda"):
    ckp = torch.load(model_file)
    model_config = edict(ckp['config'])
    model = get_score_model(config=model_config, dev=device)
    model.load_state_dict(ckp['model'], strict=False)

    model.to(device)
    model.eval()

    return model

def score_fun_edpgnn(A_tilde, U_idxs_triu, sigma_idx, model, nodes, sigmas, max_nodes):
	model.eval()

	num_sigmas = len(sigmas)
	node_flag_init = torch.tensor([1, 0], device=A_tilde.device)
	node_flags = node_flag_init.repeat_interleave(torch.tensor([nodes, max_nodes - nodes], device=A_tilde.device))
	node_flags = node_flags.repeat(num_sigmas, 1).float()
	x = torch.zeros((num_sigmas, max_nodes, 1), device=A_tilde.device).float()
	model_input = torch.zeros((num_sigmas, max_nodes, max_nodes), device=A_tilde.device).float()
	padded_A_tilde = pad_adjs(A_tilde, max_nodes)
	model_input[num_sigmas - sigma_idx - 1, :, :] = padded_A_tilde.clone().detach().float().to(A_tilde.device)

	with torch.no_grad():
		all_score_levels = model(x, model_input, node_flags)
		vectorized_selected_score = all_score_levels[num_sigmas - sigma_idx - 1, U_idxs_triu[0], U_idxs_triu[1]]
	return vectorized_selected_score.detach()
