import numpy as np
from scipy.stats import wishart


def wishart_prior(num_nodes):
    return wishart(num_nodes, np.eye(num_nodes) * 10 / num_nodes)
