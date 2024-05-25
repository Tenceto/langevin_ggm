import os, traceback, sys
import numpy as np
import pandas as pd
import torch
import logging
from functools import partial
from contextlib import redirect_stdout
from argparse import ArgumentParser
import yaml
from easydict import EasyDict as edict
from datetime import datetime
import pprint

import ggm_estimation.predictors as ggmp
import ggm_estimation.data_generation as gen
from ggm_estimation import priors
import utils.ggm_utils as ggm_utils
from utils.torch_load import load_edpgnn_from_ckpt, score_fun_edpgnn

torch.set_default_device("cuda")
torch.set_default_dtype(torch.float32)


def main(config, logger, output_file):
	logger.info(f"Running simulations for {config.score_model.model_file}.")
	logger.info(f"Config parameters used: \n{pprint.pformat(config)}")

	# Read parameters from config file

	# General parameters
	device = config.device

	# Score model parameters
	model_file = f"./trained_networks/{config.score_model.model_file}.pth"
	sigmas = np.linspace(config.score_model.sigmas.max, config.score_model.sigmas.min, config.score_model.sigmas.num)
	max_nodes = config.score_model.max_nodes
	epsilon = config.score_model.epsilon
	steps = config.score_model.steps
	temperature = config.score_model.temperature
	num_samples = config.score_model.num_samples

	# GGM simulation parameters
	obs_ratio_list = config.ggm_simulation.obs_ratio_list
	prior_Theta = getattr(priors, config.ggm_simulation.prior_Theta)
	n_proportional = config.ggm_simulation.n_proportional
	nans = config.ggm_simulation.nans
	psd_trials = config.ggm_simulation.psd_trials
	try:
		one_zero_ratio = config.ggm_simulation.one_zero_ratio
	except:
		one_zero_ratio = None
	generator_fun = getattr(gen, config.graph_generation.method)

	# GLasso parameters
	lambda_fun = partial(getattr(ggm_utils, config.lambda_glasso.fun), **config.lambda_glasso.params)

	# Define all estimators to be used
	model = load_edpgnn_from_ckpt(model_file, device=device)
	tiger = ggmp.TIGEREstimator(zero_tol=1.E-4)
	quic = ggmp.QuicEstimator(lambda_fun)
	thresholder = ggmp.QuicEstimator(lambda x: 0.0)
	# We change the default device to cpu because GraphSAGE doesn't support CUDA
	torch.set_default_device("cpu")
	sage = ggmp.GNNEstimator(h_feats=32, lr=0.005, epochs=1000)
	torch.set_default_device("cuda")

	# Start the simulations
	for seed, A in enumerate(generator_fun(**config.graph_generation.params)):
		output_results = list()
		np.random.seed(seed)

		n_missing = nans if n_proportional is False else np.ceil(nans * A.shape[0] * (A.shape[0] - 1) / 2)
		max_num_obs = int(np.ceil(obs_ratio_list[-1] * n_missing))

		try:
			A_obs, X_obs = gen.simulate_ggm(A, max_num_obs, nans, one_zero_ratio, n_proportional, 
											psd_trials, prior_Theta, logger)
		except gen.PSDGenerationError:
			logger.warning(f"Could not generate a PSD matrix. Seed: {seed}")
			continue
		
		missing_idx = np.where(np.isnan(np.triu(A_obs)))
		A_obs_torch = torch.tensor(A_obs, device=device)

		prior_A_score = partial(score_fun_edpgnn, model=model, nodes=A.shape[0], 
								sigmas=sigmas, max_nodes=max_nodes)
		
		langevin_prior = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon, 
												steps=steps, score_estimator=prior_A_score,
												use_likelihood=False, use_prior=True)
		# langevin_likelihood = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon,
		# 											   steps=steps, score_estimator=prior_A_score,
		# 											   use_likelihood=True, use_prior=False)
		langevin_posterior = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon,
													steps=steps, score_estimator=prior_A_score,
													use_likelihood=True, use_prior=True)
		
		A_langevin_prior = langevin_prior.generate_sample(A_obs_torch, X_obs=None, 
														  temperature=temperature, num_samples=1,
														  seed=seed)

		for obs_ratio in obs_ratio_list:
			num_obs = int(np.ceil(obs_ratio * n_missing))
			this_S = np.cov(X_obs[:num_obs], rowvar=False, ddof=0)

			# A_langevin_likelihood = langevin_likelihood.generate_sample(A_obs_torch, X_obs[:num_obs], 
			# 												   			temperature=temperature, num_samples=10,
			# 															seed=(seed + 1) * 3)
			A_langevin_posterior = langevin_posterior.generate_sample(A_obs_torch, X_obs[:num_obs],
															 		  temperature=temperature, num_samples=num_samples,
																	  seed=(seed + 1) * 3)
			
			A_all = {
				"langevin_posterior": A_langevin_posterior, 
				"langevin_prior": A_langevin_prior,
				# "langevin_likelihood": A_langevin_likelihood,
				"glasso": quic.generate_sample(A_obs, X_obs[:num_obs]),
				"tiger": tiger.generate_sample(A_obs, this_S, X_obs[:num_obs]),
				"threshold": thresholder.generate_sample(A_obs, X_obs[:num_obs]),
			}

			# We change the default device to cpu because GraphSAGE doesn't support CUDA
			torch.set_default_device("cpu")
			A_all.update({"gnn": sage.generate_sample(A_obs, X_obs[:num_obs])})
			torch.set_default_device("cuda")

			this_output = dict()

			this_output["seed"] = seed
			# this_output["num_obs"] = num_obs
			this_output["obs_ratio"] = obs_ratio
			# this_output["num_samples"] = num_samples
			this_output["real_values"] = A[missing_idx].tolist()
			for method, A_sampled in A_all.items():
				try:
					values_sampled = A_sampled[missing_idx].tolist()
				except:
					values_sampled = A_sampled[missing_idx].cpu().numpy().tolist()
				this_output[f"pred_{method}"] = values_sampled
			output_results.append(this_output)
		
			logger.info(f"Finished iteration. Seed: {seed}, k/|U| = {obs_ratio}, k = {num_obs}, |U|: {len(missing_idx[0])}")
		pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))

		return


if __name__ == '__main__':
	# Set up logging
	logging.basicConfig(filename=f"logs/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S.log')}",
						filemode='a',
						format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
						datefmt='%H:%M:%S',
						level=logging.DEBUG)
	logging.captureWarnings(True)
	logger = logging.getLogger('Langevin')

	def exc_handler(exctype, value, tb):
		logger.exception(''.join(traceback.format_exception(exctype, value, tb)))
	sys.excepthook = exc_handler

	# Read config file
	parser = ArgumentParser()
	parser.add_argument('-c','--config', help='Configuration file', required=True)
	args = parser.parse_args()
	config_dir = f'./config/{args.config}.yaml'
	config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))

	# Output filename
	output_file = f"outputs/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"

	with redirect_stdout(logging):
		main(config, logger, output_file)
	
	logger.info(f"Finished all iterations.")