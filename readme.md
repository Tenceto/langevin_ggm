# Estimation of partially known Gaussian graphical models with score-based structural priors

This repository contains the official source code for the paper

**Estimation of partially known Gaussian graphical models with score-based structural priors** (AISTATS 2024),

*Authors: Martín Sevilla, Antonio G. Marques and  Santiago Segarra*

The code in `scorematching_gnn` was taken from the [original repository](https://github.com/ermongroup/GraphScoreMatching) for the work **Permutation Invariant Graph Generation via Score-Based Generative Modeling** by Niu et al. (2020).

---

We propose a novel algorithm for the support estimation of partially known Gaussian graphical models that incorporates prior information about the underlying graph. 
In contrast to classical approaches that provide a *point estimate* based on a maximum likelihood or a maximum a posteriori criterion using (simple) priors *on the precision matrix*, we consider a prior *on the graph* and rely on annealed Langevin diffusion to generate *samples from the posterior distribution*.
Since the Langevin sampler requires access to the score function of the underlying graph prior, we use graph neural networks to effectively estimate the score from a graph dataset (either available beforehand or generated from a known distribution).

---

## Installation

First, (if not already installed) it is necessary to install `liblapack-dev` and `libopenblas-dev` to build the required Python package `skggm`.

`sudo apt install liblapack-dev libopenblas-dev`

Then all the Python dependencies can be installed.

`pip install -r requirements.txt`

The package `skggm` is not compatible with the last version of `sklearn`, and thus some minor (manual) changes are necessary to make the code run without errors.

Within the `skggm` installation folder:
* In `inverse_covariance/quic_graph_lasso.py`, replace `sklearn.utils.testing` by `sklearn.utils._testing`
* In `inverse_covariance/quic_graph_lasso.py` and `inverse_covariance/model_average.py`, replace `sklearn.externals.joblib` by `joblib`

To reproduce the plots in the paper, some additional packages are necessary as we use TeX fonts.
The following line is **optional** and is only necessary to *exactly* reproduce the plots:

`apt install texlive-fonts-recommended texlive-fonts-extra dvipng cm-super`


## Running experiments

There are four executable Python scripts in this repository.
We list all of them here and a quick explanation on how to use them.

### Training an EDP-GNN (`train_edpgnn.py`)
This script is exactly the same as in its [original repository](https://github.com/ermongroup/GraphScoreMatching), and the instructions can be taken from there.
In a nutshell, to fit an EDP-GNN with a given configuration file, we run

`python train_edpgnn.py -c scorematching_gnn/config/my_config_file.yaml`

### Tuning the graphical lasso (`tuning_glasso.py`)

In order to make comparisons fair, we tune the graphical lasso penalty for different types of graphs, different number of nodes, different number of missing values and different number of observations.
In particular, we find a function of the number of observations for a fixed proportion of missing values, for each type of graph (see the Supplementary Material of our paper for more information).

To run this script, just set the configuration in the first lines of the file and run it.
The output is analyzed in `notebooks/tuning_glasso.ipynb` to fit the coefficients for our function.
Once we find a fit that works fine, it has to be manually added to the function `lambda_glasso_selector()` in `ggm_estimation/utils.py` to be used during simulations.

### Experiments with partially known graphs (`main.py`)

It works similarly to the previous one.
Just adapt the parameters of the simulation at the beginning of the file, and run the script.
The output can be analyzed in `notebooks/plot_results.ipynb` by plotting the metric curves.

### Experiments with fully unknown graphs(`main_bootstrap.py`)

This script covers the case in which the graph is fully unknown and it has to be estimated altogether (as in the last experiment in Section 4 of our original paper).
Notice that in order to use this method, it is necessary to have a graphical lasso tuned beforehand to the specific simulation (i.e., with the whole graph being unknown for the corresponding type of graph prior).
The rest is the same as in `main.py`.
