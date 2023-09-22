import numpy as np
import networkx as nx
import itertools
import random
import pickle as pkl
from rpy2 import robjects

import ggm_estimation.utils as ut


def simulate_ggm(A, n_obs, nans, one_zero_ratio, n_proportional, psd_trials, prior_Theta, logger):
		num_nodes = A.shape[0]
		I = np.eye(num_nodes)

		a = ut.adj_matrix_to_vec(A)
		a_nan = ut.random_nan_replacement(a, nans, one_zero_ratio=one_zero_ratio, n_proportional=n_proportional)
		A_obs = ut.vec_to_adj_matrix(a_nan)

		trials = 0
		while trials < psd_trials:
			try:
				Theta = prior_Theta(num_nodes).rvs() * (A + I)
				# If this fails, it is because Theta is not PSD
				_ = np.linalg.cholesky(Theta)
				break
			except np.linalg.LinAlgError:
				if logger is not None:
					logger.warning(f"Not PSD matrix. Trying again.")
				pass
			trials += 1
		if trials == psd_trials:
			raise RuntimeError("Could not generate a PSD matrix.")

		inv_Theta = np.linalg.inv(Theta)
		X_obs = np.random.multivariate_normal(np.zeros(num_nodes), inv_Theta, size=n_obs)
		return A_obs, X_obs


def generate_ergm(nodes, betas, n_sim, seed_r):
    assert len(betas) == 2
    
    betas_str = ", ".join([str(theta) for theta in betas])

    string_for_r = (
        f"library(ergm)\n"
        f"f = function(){{\n"
        f"  set.seed({seed_r})\n"
        f"  n = {nodes}\n"
        f"  output = list()\n"
        f"  betas = c({betas_str})\n"
        f"      for(i in 1:{n_sim}){{\n"
        f"          g_basis = network(n, directed=FALSE)\n"
        f"          g_sim = simulate( ~ altkstar(0.3, fixed=TRUE) + edges,\n"
        f"                           nsim=1,\n"
        f"                           coef=betas,\n"
        f"                           basis=g_basis)\n"
        f"          output[[i]] = as.matrix(g_sim)\n"
        f"    }}\n"
        f"    return(output)\n"
        f"}}\n"
    )

    robjects.r(string_for_r)

    graphs = robjects.r['f']()
    graphs = [np.array(graph) for graph in graphs]

    return graphs

def load_graph_dataset(filename, n_sim):
    with open(filename, "rb") as f:
        graphs = [nx.to_numpy_array(g) for g in pkl.load(f)]
    # Randomly select n_sim graphs
    random.shuffle(graphs)
    try:
        return graphs[:n_sim]
    except IndexError:
        
        return graphs

def generate_barabasi_albert(nodes_list, m1, m2, p, n_sim, seed):
    graphs = list()
    for n in nodes_list:
        graphs.extend(_generate_barabasi_albert(n, m1, m2, p, n_sim // len(nodes_list), seed=n + seed))
    random.shuffle(graphs)

def _generate_barabasi_albert(n, m1, m2, p, n_sim, seed):
    np.random.seed(seed)

    # graphs = [nx.barabasi_albert_graph(nodes, new_edges, seed=seed + i) for i in range(n_sim)]
    graphs = [nx.dual_barabasi_albert_graph(n, m1, m2, p, seed=seed + i)
              for i in range(n_sim)]
    return [nx.to_numpy_array(g) for g in graphs]

def generate_grids(m_min, m_max, min_nodes, max_nodes, min_random_edges, max_random_edges, n_sim, seed):
    np.random.seed(seed)
    random.seed(seed + 1)

    m_values = np.arange(m_min, m_max + 1)
    possible_tuples = list(itertools.product(m_values, m_values))
    possible_tuples = [(x, y) for x, y in possible_tuples if min_nodes <= x * y <= max_nodes and x <= y]
    # print(possible_tuples)

    grid_dims = random.choices(possible_tuples, k=n_sim)
    n_random_edges = np.random.randint(min_random_edges, max_random_edges, n_sim)

    graphs = [nx.grid_2d_graph(d[0], d[1]) for d in grid_dims]
    As = [_add_random_edges(nx.to_numpy_array(g), edges) for g, edges in zip(graphs, n_random_edges)]
    return As

def _add_random_edges(A, n_added_edges):
    if n_added_edges != 0:
        zero_indices = np.argwhere(A == 0)
        zero_indices = zero_indices[zero_indices[:, 0] != zero_indices[:, 1]]

        # Choose a random zero to convert to a one
        idx = np.random.choice(np.arange(zero_indices.shape[0]), n_added_edges, replace=False)
        row, col = zero_indices[idx, 0], zero_indices[idx, 1]

        # Convert the chosen zero to a one
        A[row, col] = 1.0
        A[col, row] = 1.0
    else:
        pass

    return A
