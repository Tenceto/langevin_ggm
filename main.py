import os, traceback, sys
import numpy as np
import pandas as pd
import torch
import logging
from scipy.stats import wishart
from functools import partial
from contextlib import redirect_stdout

import ggm_estimation.predictors as ggmp
import ggm_estimation.data_generation as gen
import ggm_estimation.utils as ut
from ggm_estimation.torch_load import load_model, score_edp_wrapper

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)

logger_file = "langevin_ggm.log"
graph_type = "deezer"
seed = 1234

# Settings

# GMRF parameters
num_obs_list = [25, 100, 350, 1300]
prior_Theta = lambda num_nodes: wishart(num_nodes, np.eye(num_nodes) * 10 / num_nodes)

# Simulation parameters
n_sim = 100
nans = 0.5
one_zero_ratio = 0.5
n_proportional = True
seed = 1994 # 6732 # np.random.randint(0, 1000)
psd_trials = 10

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
	# Graphical lasso lambda
	lambda_fun = ut.lambda_generic(4.87366705, -19.5206655, 48.02892549)
	# Graph creation function
	graph_generator = partial(gen.generate_grids, 
						   	  m_min=m_min, m_max=m_max, min_nodes=min_nodes, 
						   	  max_nodes=max_nodes, min_random_edges=min_random_edges, 
						   	  max_random_edges=max_random_edges, seed=0)
elif graph_type == "ergm":
	# Graph parameters
	num_nodes = 50
	max_nodes = 50
	betas = [0.7, -2.0]
	# Prior model
	# model_file = ("scorematching_gnn/exp/ergm_aks_150/edp-gnn_ergm_aks__Feb-14-11-39-48_786992/models/" +
	# 			  "ergm_aks_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	# model_file = ("scorematching_gnn/exp/ergm_aks_250/edp-gnn_ergm_aks__Feb-18-20-35-52_3080413/models" +
	# 			  "/ergm_aks_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	# model_file = ("scorematching_gnn/exp/ergm_aks_500/edp-gnn_ergm_aks__Feb-14-15-56-46_1024772/models" +
	# 			  "/ergm_aks_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	# model_file = ("scorematching_gnn/exp/ergm_aks_300/edp-gnn_ergm_aks__Feb-14-13-40-29_860318/models/" +
	# 			  "ergm_aks_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	model_file = ("scorematching_gnn/exp/ergm_aks/edp-gnn_ergm_aks__Jan-29-11-45-37_2642242/models/erg" +
				  "m_aks_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	# Graphical lasso lambda
	lambda_fun = ut.lambda_generic(2.69135693, 4.61820941, -13.35200138)
	# Graph creation function
	graph_generator = partial(gen.generate_ergm, nodes=num_nodes, betas=betas, seed_r=seed)
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
	# Graphical lasso lambda
	lambda_fun = ut.lambda_generic(3.93966846, -6.46266204, 10.46412147)
	# Graph creation function
	graph_generator = partial(gen.generate_barabasi_albert, 
						   	  nodes_list=nodes_list, m1=m1, m2=m2, p=p, seed=seed)
elif graph_type == "deezer":
	# Graph parameters
	max_nodes = 25
	# Prior model
	model_file = ("scorematching_gnn/exp/deezer_ego/edp-gnn_train_deezer_ego__Jun-14-14-14-11_1489048/models/" +
			      "train_deezer_ego_[0.03, 0.08222222, 0.13444444, 0.18666667, 0.23888889, 0.29111111, 0.34333333, 0.39555556, 0.44777778, 0.5].pth")
	# Graphical lasso lambda
	lambda_fun = ut.lambda_generic(0.09569316, -21.02993316, 37.3193769)
	# Graph creation function
	graph_generator = partial(gen.load_graph_dataset, filename="scorematching_gnn/data/test_deezer_ego.pkl")

# Langevin predictor parameters
sigmas = np.linspace(0.5, 0.03, 10)
epsilon = 1.0E-6
steps = 300
temperature = 1.0
num_samples = 10

# Outpit file
output_file = f"results/{graph_type}_{seed}_{nans}.csv"

if __name__ == '__main__':
	logging.basicConfig(filename=f"logs/{logger_file}",
						filemode='a',
						format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
						datefmt='%H:%M:%S',
						level=logging.DEBUG)
	logger = logging.getLogger('Langevin')
	logger.info("Running Langevin Simulation")
	logging.getLogger('rpy2.rinterface_lib.callbacks').disabled = True

	def exc_handler(exctype, value, tb):
		logger.exception(''.join(traceback.format_exception(exctype, value, tb)))

	# Install exception handler
	sys.excepthook = exc_handler

	model = load_model(model_file)
	
	simulation_args = [(seed, g, logger) for seed, g in enumerate(graph_generator(n_sim=n_sim))]

	logger.info(f"Matrices generated.")

	tiger = ggmp.TIGEREstimator(zero_tol=1.E-4)
	quic = ggmp.QuicEstimator(lambda_fun)
	thresholder = ggmp.QuicEstimator(lambda x: 0.0)

	# We change the default device to cpu because GraphSAGE doesn't support CUDA
	torch.set_default_device("cpu")
	sage = ggmp.GNNEstimator(h_feats=32, lr=0.005, epochs=1000)
	torch.set_default_device("cuda")

	def simulation_wrapper(args):
		seed, A, logger = args
		output_results = list()
		np.random.seed(seed)

		A_obs, X_obs = gen.simulate_ggm(A, num_obs_list[-1], 
										nans, one_zero_ratio, n_proportional, psd_trials, prior_Theta, logger)
		missing_idx = np.where(np.isnan(np.triu(A_obs)))
		A_obs_torch = torch.tensor(A_obs)

		prior_A_score = score_edp_wrapper(model, A.shape[0], len(sigmas), max_nodes)
		
		langevin_prior = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon, 
											  	steps=steps, score_estimator=prior_A_score,
												use_likelihood=False, use_prior=True)
		langevin_likelihood = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon,
											   		 steps=steps, score_estimator=prior_A_score,
													 use_likelihood=True, use_prior=False)
		langevin_posterior = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon,
											  		steps=steps, score_estimator=prior_A_score,
													use_likelihood=True, use_prior=True)
		
		A_langevin_prior = langevin_prior.generate_sample(A_obs_torch, X_obs=None, temperature=temperature, num_samples=1)

		for num_obs in num_obs_list:
			this_S = np.cov(X_obs[:num_obs], rowvar=False, ddof=0)

			np.random.seed((seed + 1) * 3)

			A_langevin_likelihood = langevin_likelihood.generate_sample(A_obs_torch, X_obs[:num_obs], temperature=temperature, num_samples=10)
			A_langevin_posterior = langevin_posterior.generate_sample(A_obs_torch, X_obs[:num_obs], temperature=temperature, num_samples=10)
			
			A_all = {
				"langevin_posterior": A_langevin_posterior, 
				"langevin_prior": A_langevin_prior,
				"langevin_likelihood": A_langevin_likelihood,
				"glasso": quic.generate_sample(A_obs, X_obs[:num_obs]),
				"tiger": tiger.generate_sample(A_obs, this_S, X_obs[:num_obs]),
				"threshold": thresholder.generate_sample(A_obs, X_obs[:num_obs]),
			}

			# We change the default device to cpu because GraphSAGE doesn't support CUDA
			torch.set_default_device("cpu")
			A_all.update({"gnn": sage.generate_sample(A_obs, X_obs[:num_obs])})
			torch.set_default_device("cuda")

			this_output = dict()

			this_output["num_obs"] = num_obs
			# this_output["num_samples"] = num_samples
			this_output["real_values"] = A[missing_idx].tolist()
			for method, A_sampled in A_all.items():
				try:
					values_sampled = A_sampled[missing_idx].tolist()
				except:
					values_sampled = A_sampled[missing_idx].cpu().numpy().tolist()
				this_output[f"pred_{method}"] = values_sampled
			output_results.append(this_output)
		
			logger.info(f"Finished iteration.")
		pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))

		return

	# pool = Pool(cpu_cores)
	# simulation_results = pool.map(simulation_wrapper, simulation_args)
	with redirect_stdout(logging):
		for arg in simulation_args:
			try:
				simulation_wrapper(arg)
			except RuntimeError:
				logger.exception(f"Runtime error in iteration.")
				continue
	# simulation_results = [item for sublist in simulation_results for item in sublist]

	logger.info(f"Finished all iterations.")

	# pd.DataFrame(simulation_results).to_csv(output_file, sep=";")