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
	logger.info(f"Running bootstrap simulations for {config.score_model.model_file}.")
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
	quic = ggmp.QuicEstimator(lambda_fun)
	stability_selector = ggmp.StabilitySelector(config.bootstrap.n_bootstrap, lambda_fun, mode="manual", 
												n_jobs=config.n_jobs)
	
	# Start the simulations
	for seed, A in enumerate(generator_fun(**config.graph_generation.params)):
		output_results = list()
		np.random.seed(seed)

		n_missing = nans if n_proportional is False else np.ceil(nans * A.shape[0] * (A.shape[0] - 1) / 2)
		max_num_obs = int(np.ceil(obs_ratio_list[-1] * n_missing))

		try:
			_, X_obs = gen.simulate_ggm(A, max_num_obs, nans, one_zero_ratio, n_proportional, 
										psd_trials, prior_Theta, logger)
		except gen.PSDGenerationError:
			logger.warning(f"Could not generate a PSD matrix. Seed: {seed}")
			continue

		prior_A_score = partial(score_fun_edpgnn, model=model, nodes=A.shape[0], 
								sigmas=sigmas, max_nodes=max_nodes)
		
		langevin_posterior = ggmp.LangevinEstimator(sigmas=sigmas, epsilon=epsilon,
													steps=steps, score_estimator=prior_A_score,
													use_likelihood=True, use_prior=True)
		
		for obs_ratio in obs_ratio_list:
			num_obs = int(np.ceil(obs_ratio * n_missing))
			# Estimate A_obs with bootstrapped glasso
			A_stability_est = stability_selector.generate_sample(X_obs[:num_obs])
			
			A_all = {
				"glasso": quic.generate_sample(None, X_obs[:num_obs]),
				"stability": A_stability_est,
			}

			ratio_unknown = list()
			# Langevin without stability selection
			A_obs_est = torch.full(A.shape, torch.nan, device=device)
			A_obs_est = A_obs_est.fill_diagonal_(0.0)
			unknown_idx = torch.where(torch.isnan(torch.triu(A_obs_est, diagonal=1)))
			A_langevin = langevin_posterior.generate_sample(A_obs_est, X_obs[:num_obs],
															temperature=temperature, num_samples=num_samples,
															seed=(seed + 1) * 3, parallel=config.parallel, n_jobs=config.n_jobs)
			A_all[f"langevin_flat"] = A_langevin
			ratio = len(unknown_idx[0]) / (A.shape[0] * (A.shape[0] - 1) / 2)
			ratio_unknown.append(round(ratio, 2))

			# Langevin with stability selection
			for margin in config.bootstrap.margins:
				A_obs_est = stability_selector.threshold_probabilities(A_stability_est, margin=margin)
				A_obs_est = torch.tensor(A_obs_est, device=device)
				unknown_idx = torch.where(torch.isnan(torch.triu(A_obs_est, diagonal=1)))
				A_langevin = langevin_posterior.generate_sample(A_obs_est, X_obs[:num_obs],
															 	temperature=temperature, num_samples=num_samples,
															 	seed=(seed + 1) * 3, parallel=config.parallel, n_jobs=config.n_jobs)
				A_all[f"langevin_{margin}"] = A_langevin
				ratio = len(unknown_idx[0]) / (A.shape[0] * (A.shape[0] - 1) / 2)
				ratio_unknown.append(round(ratio, 2))

			triu_idx = np.triu_indices_from(A, k=1)

			this_output = dict()
			this_output["seed"] = seed
			this_output["obs_ratio"] = obs_ratio
			this_output["real_values"] = A[triu_idx].tolist()
			this_output["ratio_unknown"] = ratio_unknown

			for method, A_estimated, in A_all.items():
				try:
					values_sampled = A_estimated[triu_idx].tolist()
				except:
					values_sampled = A_estimated[triu_idx].cpu().numpy().tolist()
				this_output[f"pred_{method}"] = values_sampled
			output_results.append(this_output)

			logger.info(f"Finished iteration. Seed: {seed}, k/n = {obs_ratio}, k = {num_obs}, n = {A.shape[0]}, |U| / dim(a): {ratio_unknown}")
		pd.DataFrame(output_results).to_csv(output_file, mode='a', sep=";", header=not os.path.exists(output_file))

	return

if __name__ == '__main__':
	# Set up logging
	logging.basicConfig(filename=f"logs/bootstrap_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S.log')}",
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
	output_file = f"outputs/bootstrap_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.csv"

	with redirect_stdout(logging):
		main(config, logger, output_file)
	
	logger.info(f"Finished all iterations.")
