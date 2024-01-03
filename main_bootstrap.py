import os, traceback, sys
import numpy as np
import pandas as pd
import torch
import logging
from scipy.stats import wishart
from functools import partial
from contextlib import redirect_stdout
# import warnings

import ggm_estimation.predictors as ggmp
import ggm_estimation.data_generation as gen
import ggm_estimation.utils as ut
from ggm_estimation.torch_load import load_model, score_edp_wrapper

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)
# warnings.filterwarnings("error")

graph_type = "deezer"
logger_file = f"langevin_ggm_{graph_type}.log"

# Settings

# GMRF parameters
obs_ratio_list = [1.0, 1.8, 3.2, 5.6, 10.0]
prior_Theta = lambda num_nodes: wishart(num_nodes, np.eye(num_nodes) * 10 / num_nodes)

# Simulation parameters
n_sim = 100
seed = 1994
psd_trials = 10
margin = 0.3
n_bootstrap = 50

# Graph parameters
if graph_type == "grids":
	# Graph parameters
	m_min = 5
	m_max = 9
	min_nodes = 40
	max_nodes = 49
	min_random_edges = 2
	max_random_edges = 5
	# Prior model
	model_file = ("scorematching_gnn/exp/grids_dif_nodes/edp-gnn_grids_dif_nodes__Feb-12-22-48-21_4158697/models/" + 
				  "grids_dif_nodes_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	# Graph creation function
	graph_generator = partial(gen.generate_grids, 
						   	  m_min=m_min, m_max=m_max, min_nodes=min_nodes, 
						   	  max_nodes=max_nodes, min_random_edges=min_random_edges, 
						   	  max_random_edges=max_random_edges, seed=seed)
elif graph_type == "barabasi":
	# Graph parameters
	nodes_list = [46, 48, 50, 52]
	m1 = 2
	m2 = 4
	p = 0.5
	max_nodes = 53
	# Prior model
	model_file = ("scorematching_gnn/exp/barabasi_albert_diff_nodes/edp-gnn_barabasi_albert_[47, 49, 51, 53]__Jun-13-10-13-20_999031/models/" +
				  "barabasi_albert_[47, 49, 51, 53]_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	# Graph creation function
	graph_generator = partial(gen.generate_barabasi_albert, 
						   	  nodes_list=nodes_list, m1=m1, m2=m2, p=p, seed=seed)
elif graph_type == "deezer":
	# Graph parameters
	max_nodes = 25
	# Prior model
	model_file = ("scorematching_gnn/exp/deezer_ego/edp-gnn_train_deezer_ego__Jun-14-14-14-11_1489048/models/" +
			      "train_deezer_ego_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	# Graph creation function
	graph_generator = partial(gen.load_graph_dataset, filename="scorematching_gnn/data/test_deezer_ego.pkl")

# Langevin predictor parameters
sigmas = np.linspace(0.5, 0.03, 10)
epsilon = 1.0E-6
steps = 300
temperature = 1.0
num_samples = 10

# Output file
output_file = f"outputs/full_vs_constrained_{seed}_{graph_type}_{num_samples}_{margin}_{n_bootstrap}.csv"

# Lambda function
lambda_fun = ut.lambda_glasso_selector(graph_type, nans=1.0, nans_proportional=True, one_zero_ratio=None)

if __name__ == '__main__':
	logging.basicConfig(filename=f"logs/{logger_file}",
						filemode='a',
						format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
						datefmt='%H:%M:%S',
						level=logging.DEBUG)
	logger = logging.getLogger('Langevin')
	logger.info("Running Langevin Simulation")

	def exc_handler(exctype, value, tb):
		logger.exception(''.join(traceback.format_exception(exctype, value, tb)))

	# Install exception handler
	sys.excepthook = exc_handler

	model = load_model(model_file)
	
	simulation_args = [(seed, g, logger) for seed, g in enumerate(graph_generator(n_sim=n_sim))]

	logger.info(f"Matrices generated.")

	# tiger = ggmp.TIGEREstimator(zero_tol=1.E-4)
	quic = ggmp.QuicEstimator(lambda_fun)
	# thresholder = ggmp.QuicEstimator(lambda x: 0.0)
	stability_selector = ggmp.StabilitySelector(n_bootstrap, lambda_fun, mode="manual", n_jobs=1)

	def simulation_wrapper(args):
		seed, A, logger = args
		num_nodes = A.shape[0]
		output_results = list()
		np.random.seed(seed)

		max_num_obs = int(np.ceil(obs_ratio_list[-1] * num_nodes))

		_, X_obs = gen.simulate_ggm(A, max_num_obs, 
									nans=1.0, one_zero_ratio=None, n_proportional=True, 
									psd_trials=psd_trials, prior_Theta=prior_Theta, logger=logger)

		prior_A_score = score_edp_wrapper(model, num_nodes, len(sigmas), max_nodes)
		
		# langevin_prior = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon, 
		# 									  	steps=steps, score_estimator=prior_A_score,
		# 										use_likelihood=False, use_prior=True)
		# langevin_likelihood = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon,
		# 									   		 steps=steps, score_estimator=prior_A_score,
		# 											 use_likelihood=True, use_prior=False)
		langevin_posterior = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon,
											  		steps=steps, score_estimator=prior_A_score,
													use_likelihood=True, use_prior=True)
		
		# A_langevin_prior = langevin_prior.generate_sample(A_obs_torch, X_obs=None, 
		# 												  temperature=temperature, num_samples=1,
		# 												  seed=seed)

		for obs_ratio in obs_ratio_list:
			num_obs = int(np.ceil(obs_ratio * num_nodes))
			# this_S = np.cov(X_obs[:num_obs], rowvar=False, ddof=0)

			# Estimate A_obs with bootstrapped glasso
			A_stability_est = stability_selector.generate_sample(X_obs[:num_obs])
			A_obs_est = stability_selector.threshold_probabilities(A_stability_est, margin=margin)
			A_obs_real = A.copy()
			A_obs_real[np.isnan(A_obs_est)] = np.nan
			A_stability_real = stability_selector.generate_sample(X_obs[:num_obs], A_nan=A_obs_real)

			triu_idx = np.triu_indices_from(A, k=1)
			missing_idx = np.where(np.isnan(np.triu(A_obs_real, k=1)))

			# assert np.all(A[np.where(~ np.isnan(A_obs_real))] == A_stability_real[np.where(~ np.isnan(A_obs_real))])
			
			A_obs_est_torch = torch.tensor(A_obs_est)
			A_obs_real_torch = torch.tensor(A_obs_real)

			# A_langevin_likelihood = langevin_likelihood.generate_sample(A_obs_torch, X_obs[:num_obs], 
			# 												   			temperature=temperature, num_samples=10,
			# 															seed=(seed + 1) * 3)
			A_lpost_est = langevin_posterior.generate_sample(A_obs_est_torch, X_obs[:num_obs],
															 temperature=temperature, num_samples=num_samples,
															 seed=(seed + 1) * 3)
			A_lpost_real = langevin_posterior.generate_sample(A_obs_real_torch, X_obs[:num_obs],
													 		  temperature=temperature, num_samples=num_samples,
															  seed=(seed + 1) * 3)

			# assert torch.all(torch.Tensor(A).cuda()[torch.where(~ torch.isnan(A_obs_real_torch))] == A_lpost_real[torch.where(~ torch.isnan(A_obs_real_torch))]).item()
			
			A_all = {
				"langevin_posterior": A_lpost_est,
				"langevin_posterior_constrained": A_lpost_real,
				# "langevin_prior": A_langevin_prior,
				# "langevin_likelihood": A_langevin_likelihood,
				"glasso": quic.generate_sample(None, X_obs[:num_obs]),
				"glasso_constrained": quic.generate_sample(A_obs_real, X_obs[:num_obs]),
				# "tiger": tiger.generate_sample(A_obs, this_S, X_obs[:num_obs]),
				"stability": A_stability_est,
				"stability_constrained": A_stability_real,
				# "threshold": thresholder.generate_sample(A_obs, X_obs[:num_obs]),
			}

			this_output = dict()

			this_output["seed"] = seed
			# this_output["num_obs"] = num_obs
			this_output["obs_ratio"] = obs_ratio
			# this_output["num_samples"] = num_samples
			# this_output["real_values"] = A[triu_idx].tolist()
			this_output["real_values"] = A[missing_idx].tolist()

			for method, A_estimated, in A_all.items():
				try:
					values_sampled = A_estimated[missing_idx].tolist()
				except:
					values_sampled = A_estimated[missing_idx].cpu().numpy().tolist()
				this_output[f"pred_{method}"] = values_sampled
			output_results.append(this_output)

			ratio_unknown = round(len(missing_idx[0]) / (num_nodes * (num_nodes - 1) / 2), 2)

			logger.info(f"Finished iteration. Seed: {seed}, k/n = {obs_ratio}, k = {num_obs}, n = {num_nodes}, |U| / dim(a): {ratio_unknown}")
		pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))

		return

	with redirect_stdout(logging):
		for arg in simulation_args:
			try:
				simulation_wrapper(arg)
			except RuntimeError:
				logger.exception(f"Runtime error in iteration.")
				continue
		# pool = Pool(5)
		# simulation_results = pool.map(simulation_wrapper, simulation_args)

	logger.info(f"Finished all iterations.")

	# pd.DataFrame(simulation_results).to_csv(output_file, sep=";")