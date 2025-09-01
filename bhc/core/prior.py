# -*- coding: utf-8 -*-

# License: GPL 3.0

import numpy as np
from scipy.special import multigammaln

from bhc.api import AbstractPrior

LOG2PI = np.log(2 * np.pi)
LOG2 = np.log(2)


class NormalInverseWishart(AbstractPrior):
    """
    Reference: MURPHY, Kevin P.
               Conjugate Bayesian analysis of the Gaussian distribution.
               def, v. 1, n. 2σ2, p. 16, 2007.
               https://www.cse.iitk.ac.in/users/piyush/courses/
               tpmi_winter19/readings/bayesGauss.pdf
    """

    def __init__(self, s_mat, r, v, m):
        self.s_mat = s_mat
        self.r = r
        self.v = v
        self.m = m
        self.log_prior0 = NormalInverseWishart.__calc_log_prior(s_mat, r, v)

    def calc_log_mlh(self, x_mat):
        x_mat_l = x_mat.copy()
        x_mat_l = x_mat_l[np.newaxis] if x_mat_l.ndim == 1 else x_mat_l
        n, d = x_mat_l.shape
        s_mat_p, rp, vp = NormalInverseWishart.__calc_posterior(
            x_mat_l, self.s_mat, self.r, self.v, self.m
        )
        log_prior = NormalInverseWishart.__calc_log_prior(s_mat_p, rp, vp)
        return log_prior - self.log_prior0 - LOG2PI * (n * d / 2.0)

    def row_of_log_likelihood_for_pairs(
        self,
        X,  # (N, d) data matrix
        i,  # index of the row you want (int)
    ):
        """
        Returns 1D array containing the log-likelihoods for pairs of points
        needed for the initialization of bhc. This function combines i with
        all other points j > i and returns the log-likelihood of those
        clusters (containing two points each).
        """
        N, d = X.shape
        if d != self.s_mat.shape[0]:
            raise ValueError(
                "data dimension and prior scale matrix do not match"
            )

        # ------------------------------------------------------------------
        # Pairwise sufficient statistics – only for j > i (batched)
        # ------------------------------------------------------------------
        # slice of points that matter
        Xj = X[i + 1:]  # shape (N-i-1, d)
        diff = X[i] - Xj  # broadcasted automatically
        x_bar = 0.5 * (X[i] + Xj)  # (N-i-1, d)

        # Scatter matrix S = ½ diff·diffᵀ  → (N-i-1, d, d)
        S = 0.5 * np.einsum("...i,...j->...ij", diff, diff)
        # Term (r·2/(r+2))·(x̄‑m)(x̄‑m)ᵀ
        dt = x_bar - self.m  # (N-i-1, d)
        outer_dt = np.einsum("...i,...j->...ij", dt, dt)  # (N-i-1, d, d)
        term = (self.r * 2.0 / (self.r + 2.0)) * outer_dt
        # Posterior scale matrix for each pair
        s_mat_p = self.s_mat[None, :, :] + S + term  # (N-i-1, d, d)

        # ------------------------------------------------------------------
        # Log‑posterior for each pair (batched)
        # ------------------------------------------------------------------
        rp = self.r + 2.0  # each cluster has two points
        vp = self.v + 2.0
        sign, logdet = np.linalg.slogdet(s_mat_p)  # (N-i-1,)
        log_prior_post = (
            LOG2 * (vp * d / 2.0)
            + (d / 2.0) * np.log(2.0 * np.pi / rp)
            + multigammaln(vp / 2.0, d)
            - (vp / 2.0) * logdet
        )  # (N-i-1,)

        # Final log-likelihood for each pair
        return log_prior_post - self.log_prior0 - LOG2PI * d  # (N-i-1,)

    @staticmethod
    def __calc_log_prior(s_mat, r, v):
        d = s_mat.shape[0]
        log_prior = LOG2 * (v * d / 2.0) + (d / 2.0) * np.log(2.0 * np.pi / r)
        log_prior += multigammaln(v / 2.0, d) - (v / 2.0) * np.log(
            np.linalg.det(s_mat)
        )
        return log_prior

    @staticmethod
    def __calc_posterior(x_mat, s_mat, r, v, m):
        n = x_mat.shape[0]
        x_bar = np.mean(x_mat, axis=0)
        rp = r + n
        vp = v + n
        s_mat_t = (
            np.zeros(s_mat.shape) if n == 1 else (n - 1) * np.cov(x_mat.T)
        )
        dt = (x_bar - m)[np.newaxis]
        s_mat_p = s_mat + s_mat_t + (r * n / rp) * np.dot(dt.T, dt)
        return s_mat_p, rp, vp

    @staticmethod
    def create(data, g, scale_factor):
        degrees_of_freedom = data.shape[1] + 1
        data_mean = np.mean(data, axis=0)
        data_matrix_cov = np.cov(data.T)
        scatter_matrix = (data_matrix_cov / g).T

        return NormalInverseWishart(
            scatter_matrix, scale_factor, degrees_of_freedom, data_mean
        )
