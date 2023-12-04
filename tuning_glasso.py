import os, traceback, sys
import numpy as np
import pandas as pd
from functools import partial
import logging
from scipy.stats import wishart
import pickle as pkl
import networkx as nx
from sklearn.metrics import f1_score, accuracy_score

from inverse_covariance import QuicGraphicalLasso
import ggm_estimation.data_generation as gen


logger_file = "tuning_glasso.log"
graph_type = "deezer"
n_sim = 200
nans = 1.0
one_zero_ratio = None
n_proportional = True
metric = "f1"

# Simulation parameters
seed = 400
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
	graph_generator = partial(gen.generate_grids, 
						   	  m_min=m_min, m_max=m_max, min_nodes=min_nodes, 
						   	  max_nodes=max_nodes, min_random_edges=min_random_edges, 
						   	  max_random_edges=max_random_edges, seed=0)
elif graph_type == "ergm":
	# Graph parameters
	num_nodes = 50
	max_nodes = 50
	betas = [0.7, -2.0]
	graph_generator = partial(gen.generate_ergm, nodes=num_nodes, betas=betas, seed_r=seed)
elif graph_type == "barabasi":
	# Graph parameters
	nodes_list = [46, 48, 50, 52]
	m1 = 2
	m2 = 4
	p = 0.5
	max_nodes = 53
	graph_generator = partial(gen.generate_barabasi_albert, 
						   	  nodes_list=nodes_list, m1=m1, m2=m2, p=p, seed=seed)
elif graph_type == "deezer":
	graph_generator = partial(gen.load_graph_dataset, filename="scorematching_gnn/data/test_deezer_ego.pkl")

if metric == "f1":
	metric_fun = f1_score
elif metric == "accuracy":
	metric_fun = accuracy_score
else:
	raise ValueError("Invalid metric.")

# GMRF parameters
prior_Theta = lambda num_nodes: wishart(num_nodes, np.eye(num_nodes) * 10 / num_nodes)

# Tuning grid parameters
lambda_inf = 10000
num_obs_list = np.logspace(np.log10(25), np.log10(1500), 6).astype(int)
lambdas = np.logspace(-3, -1, 30)

output_file = f"outputs/tuning/tuning_glasso_{graph_type}_{nans}_{n_proportional}_{one_zero_ratio}_{metric}.csv"

# Run your main script here:
if __name__ == '__main__':
	logging.basicConfig(filename=f"logs/{logger_file}",
						filemode='a',
						format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
						datefmt='%H:%M:%S',
						level=logging.DEBUG)
	logger = logging.getLogger('Langevin')
	logger.info("Running GLasso tuning analysis.")
	logging.getLogger('rpy2.rinterface_lib.callbacks').disabled = True

	def exc_handler(exctype, value, tb):
		logger.exception(''.join(traceback.format_exception(exctype, value, tb)))

	# Install exception handler
	sys.excepthook = exc_handler

	graphs = graph_generator(n_sim=n_sim)

	logger.info(f"Matrices generated.")

	for A in graphs:
		output_results = []
		try:
			A_obs, all_X = gen.simulate_ggm(A, num_obs_list[-1], 
											nans, one_zero_ratio, n_proportional, psd_trials, prior_Theta, logger)
		except RuntimeError:
			logger.warning(f"Could not generate a PSD matrix. Skipping.")
			continue
		missing_idx = np.where(np.isnan(np.triu(A_obs)))

		diag_idxs = np.diag_indices_from(A_obs)
		mask_inf_penalty = A_obs == 0
		mask_inf_penalty[diag_idxs] = False
		mask_unknown = np.isnan(A_obs)

		Lambda = np.zeros(A_obs.shape)
		# The "infinite penalty" should not be ridiculously high
		# Otherwise the algorithm becomes numerically unstable
		# (Before I was using np.inf and it wasn't working properly)
		Lambda[mask_inf_penalty] = lambda_inf

		for num_obs in num_obs_list:
			X_obs = all_X[:num_obs]
			for lam in lambdas:
				Lambda[mask_unknown] = lam
				model = QuicGraphicalLasso(lam=Lambda, init_method="cov", auto_scale=False)
				Theta_quic = model.fit(X_obs).precision_
				A_quic = (np.abs(Theta_quic - np.diag(np.diag(Theta_quic))) != 0.0).astype(float)
				metric_score = metric_fun(A[missing_idx], A_quic[missing_idx])
		
				output_results.append({"num_obs": num_obs, "lambda": lam, "metric": metric_score})
		
		pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))
		logger.info(f"Finished iteration. Removed 0: {np.sum(A[missing_idx] == 0)}, Removed 1: {np.sum(A[missing_idx] == 1)}.")

	logger.info(f"Finished all iterations.")

	# pd.DataFrame(simulation_results).to_csv(output_file, sep=";")
