import os, traceback, sys
import numpy as np
import pandas as pd
from functools import partial
import logging
from scipy.stats import wishart
import torch
import random
import pickle as pkl
import networkx as nx
from contextlib import redirect_stdout
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

# os.chdir("/home/msevilla/link_prediction/")

import ggm_estimation.ergm.statistics as st
import ggm_estimation.ergm.tools as et
import ggm_estimation.grids.tools as gr
import ggm_estimation.barabasi.tools as ba
from inverse_covariance import QuicGraphicalLasso
import ggm_estimation.utils as mtx

logger_file = "tuning_glasso.log"
output_file = "outputs/tuning_glasso_deezer_all_missing.csv"

# Settings
# Graph parameters

# num_nodes = 50
# thetas = [0.7, -2.0]
# global_statistics_r = [f"altkstar({st.LAMBDA_}, fixed=TRUE)", "edges"]
# global_statistics_py = ["altkstar", "edges"]

# m_min = 5
# m_max = 9
# min_nodes = 40
# max_nodes = 49
# min_random_edges = 2
# max_random_edges = 5

# nodes = [47, 49, 51, 53]
# m1 = 2
# m2 = 4
# p = 0.5
# n_sim = 1000 // len(nodes)

# GMRF parameters
# prior_Theta = wishart(num_nodes, np.eye(num_nodes) * 10 / num_nodes)

prior_Theta = lambda num_nodes: wishart(num_nodes, np.eye(num_nodes) * 10 / num_nodes)

# prior_Theta = wishart(nodes, np.eye(nodes) * 10 / nodes)

# Simulation parameters
n_sim = 200
nans = 1.0
one_zero_ratio = 0.5
n_proportional = True
seed = 400 # np.random.randint(0, 1000)
psd_trials = 10

# Tuning grid parameters
lambda_inf = 10000
num_obs_list = np.logspace(np.log10(25), np.log10(1500), 6).astype(int)
lambdas = np.logspace(-3, -1, 30)

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

	# graphs = et.generate_matrices(num_nodes, thetas, n_sim, global_statistics_r, num_attr=0, seed_r=seed)

	# graphs = gr.generate_matrices(m_min, m_max, min_nodes, max_nodes, min_random_edges, max_random_edges, n_sim, seed=seed)

	# graphs = ba.generate_matrices(nodes, m1, m2, p, n_sim, seed=seed)
	# graphs = list()
	# for n in nodes:
	# 	graphs.extend(ba.generate_matrices(n, m1, m2, p, n_sim, seed=n))
	# random.shuffle(graphs)

	with open("scorematching_gnn/data/test_deezer_ego.pkl", "rb") as f:
		graphs = pkl.load(f)
	graphs = [nx.to_numpy_array(g) for g in graphs]

	# I = np.eye(num_nodes)

	def simulate_data(A, n_obs):
		num_nodes = A.shape[0]
		I = np.eye(num_nodes)

		a = mtx.adj_matrix_to_vec(A)
		a_nan = mtx.random_nan_replacement(a, nans, one_zero_ratio=one_zero_ratio, n_proportional=n_proportional)
		A_obs = mtx.vec_to_adj_matrix(a_nan)

		trials = 0
		while trials < psd_trials:
			try:
				# Theta = prior_Theta.rvs() * (A + I)
				Theta = prior_Theta(num_nodes).rvs() * (A + I)
				# If this fails, it is because Theta is not PSD
				_ = np.linalg.cholesky(Theta)
				break
			except np.linalg.LinAlgError:
				print(f"Not PSD matrix. Trying again.")
				pass
			trials += 1
		if trials == psd_trials:
			raise RuntimeError("Could not generate a PSD matrix.")

		inv_Theta = np.linalg.inv(Theta)
		X_obs = np.random.multivariate_normal(np.zeros(num_nodes), inv_Theta, size=n_obs)
		return A_obs, X_obs, Theta

	logger.info(f"Matrices generated.")

	for g in graphs:
		output_results = []

		try:
			A = g["A"]
		except IndexError:
			A = g
		try:
			A_obs, all_X, Theta = simulate_data(A, num_obs_list[-1])
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
				f1 = f1_score(A[missing_idx], A_quic[missing_idx])
				auc = roc_auc_score(A[missing_idx], A_quic[missing_idx])
		
				output_results.append({"num_obs": num_obs, "lambda": lam, "f1": f1, "auc": auc})
		
		pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))
		logger.info(f"Finished iteration.")

	logger.info(f"Finished all iterations.")

	# pd.DataFrame(simulation_results).to_csv(output_file, sep=";")