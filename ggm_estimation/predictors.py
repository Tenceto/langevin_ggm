import numpy as np
import cvxpy as cp
import torch
from torch import nn
import itertools
from inverse_covariance import QuicGraphicalLasso, ModelAverage
from dgl.nn import SAGEConv
import torch.nn.functional as F
import dgl


class LangevinEstimator:
    def __init__(self, sigmas, epsilon, steps, score_estimator, use_prior=True, use_likelihood=True):
        self.sigmas_sq = sigmas ** 2
        self.epsilon = epsilon
        self.steps = steps
        self.score_estimator = score_estimator
        self.use_prior = use_prior
        self.use_likelihood = use_likelihood

    def _compute_raw_theta(self, A_nan, X_obs):
        diag_idxs = np.diag_indices_from(A_nan)
        mask_inf_penalty = A_nan == 0
        mask_inf_penalty[diag_idxs] = False

        Lambda = np.zeros(A_nan.shape)
        Lambda[mask_inf_penalty] = 10000

        model_cv = QuicGraphicalLasso(lam=Lambda, init_method="cov")
        Theta_quic = model_cv.fit(X_obs).precision_

        return Theta_quic
    
    def generate_sample(self, A_nan, X_obs, temperature=1.0, num_samples=1, seed=None, levels_no_prior=0):
        U_idxs_triu = torch.where(torch.isnan(torch.triu(A_nan)))
        O_mask = ~ torch.isnan(A_nan)
        if self.use_likelihood:
            # Sample covariance
            S = torch.tensor(np.cov(X_obs, rowvar=False, ddof=0))
            # Raw estimator for Theta
            Theta_est = torch.tensor(self._compute_raw_theta(A_nan.cpu().numpy(), X_obs))
            # Number of observations (k)
            num_obs = X_obs.shape[0]
        else:
            S, Theta_est, num_obs = None, None, 0
        
        As = []
        for m in range(num_samples):
            if seed is not None:
                torch.manual_seed(seed + m)
                np.random.seed(seed + m)
            this_A = self._generate_individual_sample(A_nan, S, Theta_est, temperature, levels_no_prior, U_idxs_triu, O_mask, num_obs)
            As.append(this_A)
        A = torch.stack(As).mean(dim=0)
        return A

    def _generate_individual_sample(self, A_nan, S, Theta_est, temperature, levels_no_prior, U_idxs_triu, O_mask, num_obs):
        size_U = len(U_idxs_triu[0])
        I = torch.eye(A_nan.shape[0])
        z_dist = torch.distributions.MultivariateNormal(torch.zeros(size_U), torch.eye(size_U))

        A_tilde = torch.distributions.Normal(0.5, 0.5).sample(A_nan.shape)
        A_tilde = torch.tril(A_tilde) + torch.tril(A_tilde, -1).T
        A_tilde.fill_diagonal_(0.0)
        A_tilde[O_mask] = A_nan.float()[O_mask]

        for sigma_i_idx, sigma_i_sq in enumerate(self.sigmas_sq):
            alpha = self.epsilon * sigma_i_sq / self.sigmas_sq[-1]
            sigma_i = np.sqrt(sigma_i_sq)
            use_prior = self.use_prior #and levels_no_prior >= sigma_i_idx

            for _ in range(self.steps):
                z = z_dist.sample([1])

                if use_prior:
                    score_prior = self.score_estimator(A_tilde, U_idxs_triu, sigma_idx=sigma_i_idx)
                else:
                    score_prior = 0.0
                if self.use_likelihood:
                    cov_inv = Theta_est * (A_tilde + I)
                    cov = torch.linalg.inv(cov_inv)
                    aux_matrix = - torch.diag(torch.diag(cov)) + 2 * cov - 2 * S + torch.diag(torch.diag(S))
                    score_likelihood = (num_obs * aux_matrix[U_idxs_triu] * Theta_est[U_idxs_triu] * 0.5).float()
                else:
                    score_likelihood = 0.0

                delta = score_prior + score_likelihood
                A_tilde = self._update_matrix(A_tilde, U_idxs_triu, alpha, delta, z, sigma_i, temperature)

        A_proj = torch.clip(A_tilde, 0.0, 1.0).round()

        return A_proj

    def _update_matrix(self, A_tilde, U_idxs_triu, alpha, delta, z, sigma_i, temperature):
        # prev_A_tilde = A_tilde.copy()
        A_tilde[U_idxs_triu[0], U_idxs_triu[1]] = (A_tilde[U_idxs_triu[0], U_idxs_triu[1]]
                                                    + alpha * delta + torch.sqrt(torch.tensor(2 * alpha * temperature)) * z)
        A_tilde[U_idxs_triu[1], U_idxs_triu[0]] = A_tilde[U_idxs_triu[0], U_idxs_triu[1]]

        # If there are NaN values it's because A_tilde is way too high or too low
        # So we clip the previous values
        # nan_idxs = np.isnan(A_tilde)
        # if np.any(nan_idxs):
        #     min_clip, max_clip = 0.0 - sigma_i, 1.0 + sigma_i
        #     A_tilde[nan_idxs] = np.clip(prev_A_tilde[nan_idxs], min_clip, max_clip)

        return A_tilde

class StabilitySelector:
    def __init__(self, n_bootstrap, lambda_fun, mode="manual", n_jobs=1):
        self.n_bootstrap = n_bootstrap
        self.mode = mode
        self.lambda_fun = lambda_fun
        self.n_jobs = n_jobs
    
    def generate_sample(self, X_obs):
        num_obs = X_obs.shape[0]
        lam = self.lambda_fun(num_obs)

        if self.mode == "manual":
            model = QuicGraphicalLasso(lam=lam, init_method="cov", auto_scale=False, verbose=False)
            supp_probs = list()
            for i in range(self.n_bootstrap):
                X_bootstrap = X_obs[np.random.choice(num_obs, num_obs, replace=True)]
                model.fit(X_bootstrap)
                supp_probs.append(model.precision_ != 0)
            # Take the mean
            supp_probs = np.mean(supp_probs, axis=0)
        elif self.mode == "auto":
            model = ModelAverage(
                estimator=QuicGraphicalLasso(lam=lam, init_method="cov", auto_scale=False, verbose=False),
                n_trials=self.n_bootstrap,
                penalization="subsampling",
                support_thresh=0.5,
                n_jobs=self.n_jobs
            )
            model.fit(X_obs)
            supp_probs = model.proportion_

        return supp_probs

    def predict_fixed(self, X_obs, margin):
        pred = self.generate_sample(X_obs)
        pred = self.threshold_probabilities(pred, margin)
        return pred

    @staticmethod
    def threshold_probabilities(probs, margin):
        pred = probs.copy()
        pred[pred >= (0.5 + margin)] = 1
        pred[pred <= (0.5 - margin)] = 0
        pred[(pred != 0) & (pred != 1)] = np.nan
        return pred


class QuicEstimator:
    def __init__(self, lambda_fun, lambda_inf=10000):
        self.lambda_fun = lambda_fun
        self.lambda_inf = lambda_inf
    
    def generate_sample(self, A_nan, X_obs):
        diag_idxs = np.diag_indices_from(A_nan)
        mask_inf_penalty = A_nan == 0
        mask_inf_penalty[diag_idxs] = False
        mask_unknown = np.isnan(A_nan)

        Lambda = np.zeros(A_nan.shape)
        lambda_n = self.lambda_fun(X_obs.shape[0])
        # The "infinite penalty" should not be ridiculously high
        # Otherwise the algorithm becomes numerically unstable
        # (Before I was using np.inf and it wasn't working properly)
        Lambda[mask_inf_penalty] = self.lambda_inf
        Lambda[mask_unknown] = lambda_n
        model = QuicGraphicalLasso(lam=Lambda, init_method="cov", auto_scale=False)
        Theta_quic = model.fit(X_obs).precision_

        if lambda_n > 0:
            A_quic = (np.abs(Theta_quic - np.diag(np.diag(Theta_quic))) != 0.0).astype(float)
        else:
            A_quic = np.abs(Theta_quic - np.diag(np.diag(Theta_quic)))

        return A_quic


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']


class GNNEstimator:
    def __init__(self, h_feats, lr, epochs):
        self.h_feats = h_feats
        self.predictor =  MLPPredictor(h_feats)
        self.lr = lr
        self.epochs = epochs

    def generate_sample(self, A_nan, X_obs):
        nodes = A_nan.shape[0]
        train_pos_u, train_pos_v = np.where(A_nan == 1)
        train_neg_u, train_neg_v = np.where((A_nan + np.eye(nodes)) == 0)

        train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=nodes)
        train_neg_g = dgl.graph((train_neg_u, train_neg_v), num_nodes=nodes)

        train_g = dgl.graph(np.nonzero(np.nan_to_num(A_nan, copy=True, nan=0.0)), num_nodes=nodes)
        train_g.ndata['feat'] = torch.tensor(X_obs.T).float()

        model = GraphSAGE(train_g.ndata['feat'].shape[1], self.h_feats)
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), self.predictor.parameters()),
                                     lr=self.lr)

        for e in range(self.epochs):
            h = model(train_g, train_g.ndata['feat'])
            pos_score = self.predictor(train_pos_g, h)
            neg_score = self.predictor(train_neg_g, h)
            loss = self._compute_loss(pos_score, neg_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        test_g = dgl.graph(np.where(np.isnan((A_nan))), num_nodes=nodes)
        with torch.no_grad():
            pred_scores = self.predictor(test_g, h).numpy()
            pred_scores = (pred_scores - pred_scores.min()) / (pred_scores.max() - pred_scores.min())
        
        A_est = A_nan.copy()
        A_est[test_g.edges()[0], test_g.edges()[1]] = pred_scores
        A_est = (A_est + A_est.T) / 2

        return A_est

    def _compute_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        return F.binary_cross_entropy_with_logits(scores, labels)


class TIGEREstimator:
    def __init__(self, zero_tol):
        self.zero_tol = zero_tol

    def generate_sample(self, A_nan, S, X_obs):
        nodes = A_nan.shape[0]
        n_obs = X_obs.shape[0]
        A_mask = (1 - ((A_nan + np.eye(nodes)) == 0).astype(int))

        Gamma_inv_sqrt = np.diag(np.diag(S) ** (-0.5))
        n_inv_sqrt = 1 / np.sqrt(n_obs)
        ksi = np.sqrt(2) / np.pi
        lam = ksi * np.pi * np.sqrt(np.log(nodes) / (2 * n_obs))
        Z_obs = X_obs @ Gamma_inv_sqrt

        Theta_cols = list()

        for j in range(nodes):
            beta = cp.Variable((nodes - 1, 1))
            # beta_constrained = cp.multiply(beta, A_mask[:, p].reshape(-1, 1))
            Z_without_col_j = np.delete(Z_obs, j, axis=1)
            obj = cp.Minimize(n_inv_sqrt * cp.pnorm(Z_obs[:, j].reshape(-1, 1) - Z_without_col_j @ beta, 2) + lam * cp.pnorm(beta, 1))
            constraints = []
            prob = cp.Problem(obj, constraints)
            prob.solve()

            beta = beta.value
            tau = n_inv_sqrt * np.linalg.norm(Z_obs[:, j].reshape(-1, 1) - Z_without_col_j @ beta)
            col_j = - (tau ** -2) * Gamma_inv_sqrt[j, j] * np.delete(np.delete(Gamma_inv_sqrt, j, axis=0), j, axis=1) @ beta
            col_j = np.insert(col_j, j, tau ** -2 * Gamma_inv_sqrt[j, j]).reshape(-1, 1)
            Theta_cols.append(col_j)

        Theta_est_tiger = self._make_symmetric(np.hstack(Theta_cols), A_mask)
        A_est_tiger = ((Theta_est_tiger - np.diag(np.diag(Theta_est_tiger))) != 0).astype(int)

        return A_est_tiger

    def _make_symmetric(self, Omega_est, A_mask):
        Omega_est_T = Omega_est.T
        mask = np.less(np.abs(Omega_est), np.abs(Omega_est_T))
        Omega_est = np.where(mask, Omega_est, Omega_est_T)

        Theta_est = Omega_est * A_mask
        # zero_tol = np.abs(Theta_est[A_nan == 1]).min()
        Theta_est[np.abs(Theta_est) < self.zero_tol] = 0.0

        return Theta_est
