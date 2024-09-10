"""
Defines the MixtureSampler class, which samples from the various distributions of our model.
"""
from typing import Dict
import numpy as np
import pandas as pd
from scipy.linalg import block_diag


class MixtureSampler:
    """
    Allows sampling from distributions using a common np.random.Generator object.
    """
    _generator: np.random.Generator
    _prior_parameters: Dict
    _N: int
    _X: np.ndarray
    _y: np.ndarray

    def __init__(self, generator: np.random.Generator, prior_parameters: Dict, independent_variables: np.ndarray,
                 dependent_variables: np.ndarray):
        self._generator = generator
        self._prior_parameters = prior_parameters
        self._N = len(independent_variables)
        self._X = independent_variables
        self._y = dependent_variables

    def sample_from_posterior(self, num_samples: int = 1):
        draws_of_h = [self._h_prior_distribution(num_samples=1)]
        draws_of_alpha_js = [self._alpha_prior_distribution()]
        draws_of_h_js = [self._h_j_prior_distribution(num_samples=self._prior_parameters["m"])]
        draws_of_s_is = [self._generator.choice(self._prior_parameters["m"], len(self._X))]
        draws_of_mu_js = [self._mu_j_prior_distribution(num_samples=self._prior_parameters["m"])]
        draws_of_betas = [self._beta_prior_distribution(num_samples=1)]

        for i in range(1, num_samples):
            # h block
            current_h = self._h_block_distribution(draws_of_h_js[-1], draws_of_s_is[-1], draws_of_mu_js[-1],
                                                   draws_of_betas[-1])
            draws_of_h.append(current_h)

            # draws_of_alpha_js block
            current_alpha_js = self._alphas_block_distribution(draws_of_s_is[-1])
            draws_of_alpha_js.append(current_alpha_js)

            # h_j block
            current_h_js = self._h_j_block_distribution(draws_of_s_is[-1], draws_of_h[-1], draws_of_betas[-1],
                                                        draws_of_mu_js[-1])
            draws_of_h_js.append(current_h_js)

            # s_i mixture assignments block
            current_s_is = self._s_i_equals_j_block(draws_of_alpha_js[-1],
                                                    draws_of_h_js[-1],
                                                    draws_of_h[-1],
                                                    draws_of_mu_js[-1],
                                                    draws_of_s_is[-1],
                                                    draws_of_betas[-1])
            draws_of_s_is.append(current_s_is)

            # mu_j and beta block
            current_mu_js, current_betas = self._gamma_block_distribution(draws_of_h[-1], draws_of_s_is[-1],
                                                                          draws_of_h_js[-1])
            draws_of_mu_js.append(current_mu_js)
            draws_of_betas.append(current_betas)

        if num_samples == 1:
            return draws_of_betas[0], draws_of_h[0], draws_of_alpha_js[0], draws_of_h_js[0], draws_of_mu_js[0], \
            draws_of_s_is[0]
        else:
            # Currently, we have one Python list of samples for each unobservable.
            samples_as_lists = [draws_of_betas, draws_of_h, draws_of_alpha_js, draws_of_h_js, draws_of_mu_js,
                                draws_of_s_is]

            samples_as_arrays = []
            # We will np.vstack each of those lists so that we have one NumPy array of samples for each unobservable.
            for sample in samples_as_lists:
                samples_as_arrays.append(np.vstack(sample))
            return samples_as_arrays

    def sample_from_prior(self):
        betas = self._beta_prior_distribution()
        h = self._h_j_prior_distribution()
        alpha = self._alpha_prior_distribution()
        h_js = self._h_j_prior_distribution(num_samples=self._prior_parameters["m"])
        mu_js = self._mu_j_prior_distribution(num_samples=self._prior_parameters["m"])
        return betas, h, alpha, h_js, mu_js

    def _beta_prior_distribution(self, num_samples: int = 1):
        """
        Sample from a multivariate normal prior distribution over beta.
        :param beta_underbar: Mean of prior distribution.
        :param H_underbar: Precision of prior distribution; inverse of variance.
        :param num_samples: Number of samples.
        :return: A sample of the requested size.
        """
        mean = self._prior_parameters["beta_underbar"]
        variance = np.linalg.inv(self._prior_parameters["H_underbar"])
        return self._generator.multivariate_normal(mean, variance, num_samples, check_valid='raise')

    def _h_prior_distribution(self, num_samples: int = 1):
        """
        Sample from a gamma prior on h.
        :param nu_underbar: Alpha parameter of gamma distribution.
        :param s2_underbar: Beta parameter of prior distribution.
        :param num_samples: Number of samples.
        :return: A sample of the requested size.
        """
        alpha = self._prior_parameters["nu_underbar"] / 2
        beta = self._prior_parameters["s2_underbar"] / 2
        return self._generator.gamma(alpha, beta, num_samples)

    def _alpha_prior_distribution(self):
        alpha = self._prior_parameters["alpha_underbar"]
        return self._generator.dirichlet(alpha)

    def _h_j_prior_distribution(self, num_samples: int = 1):
        """
        Sample from a gamma prior on h.
        :param nu_j_underbar: Alpha parameter of gamma distribution.
        :param s2_j_underbar: Beta parameter of prior distribution.
        :param num_samples: Number of samples.
        :return: A sample of the requested size.
        """
        alpha = self._prior_parameters["nu_j_underbar"] / 2
        beta = self._prior_parameters["s2_j_underbar"] / 2
        return self._generator.gamma(alpha, beta, num_samples)

    def _mu_j_prior_distribution(self, num_samples: int = 1):
        mean = self._prior_parameters["mu_j_underbar"]
        variance = 1 / (self._prior_parameters["h_underbar"] * self._prior_parameters["h_mu_underbar"])
        return self._generator.normal(mean, variance, num_samples)

    def _h_block_distribution(self,
                              most_recent_h_j: np.ndarray,
                              most_recent_s_i_equals_j: np.ndarray,
                              most_recent_mu_j: np.ndarray,
                              most_recent_beta: np.ndarray):
        nu_overbar = self._prior_parameters["nu_underbar"] + self._prior_parameters["m"] + self._N
        s_2_overbar = (self._prior_parameters["s2_underbar"] +
                       self._prior_parameters["h_mu_underbar"] * np.sum(np.power(most_recent_mu_j, 2)) +
                       np.sum(most_recent_h_j[most_recent_s_i_equals_j] * np.power(
                           self._y - most_recent_mu_j[most_recent_s_i_equals_j] - most_recent_beta @ self._X.T, 2))
                       )
        # Draw from block.
        alpha = nu_overbar / 2
        beta = s_2_overbar / 2
        return self._generator.gamma(alpha, 1 / beta)

    def _alphas_block_distribution(self, most_recent_s_i_equals_j: np.ndarray):
        # Calculate block parameters.

        counts = []
        for component in range(self._prior_parameters["m"]):
            counts.append(np.sum(most_recent_s_i_equals_j == component))
        counts = np.array(counts)

        alpha_overbar = (self._prior_parameters["alpha_underbar"] + counts)

        return self._generator.dirichlet(alpha_overbar)

    def _h_j_block_distribution(self,
                                most_recent_s_i_equals_j: np.ndarray,
                                most_recent_h: np.ndarray,
                                most_recent_beta: np.ndarray,
                                most_recent_mu_j: np.ndarray):
        h_j_draw = np.zeros(self._prior_parameters["m"])
        for j in range(self._prior_parameters["m"]):
            nu_j_overbar = self._prior_parameters["nu_j_underbar"] + np.sum(most_recent_s_i_equals_j == j)
            s_2_j_overbar = (self._prior_parameters["nu_j_underbar"] +
                             most_recent_h *
                             np.sum((most_recent_s_i_equals_j == j) * np.power(
                                 self._y - most_recent_mu_j[most_recent_s_i_equals_j] - most_recent_beta @ self._X.T,
                                 2)))
            alpha = nu_j_overbar / 2
            beta = s_2_j_overbar / 2
            h_j_draw[j] = self._generator.gamma(alpha, 1 / beta)
        return h_j_draw

    def _s_i_equals_j_block(self,
                            most_recent_alpha: np.ndarray,
                            most_recent_h_j: np.ndarray,
                            most_recent_h: np.ndarray,
                            most_recent_mu_j: np.ndarray,
                            most_recent_s_i_equals_j: np.ndarray,
                            most_recent_beta: np.ndarray):
        s_i_equals_j = []
        for i in range(len(self._X)):
            probabilities = (most_recent_alpha *
                             np.power(most_recent_h_j, 1 / 2) *
                             np.exp(-1 * most_recent_h * most_recent_h_j * np.power(
                                 self._y[i] - most_recent_mu_j - most_recent_beta @ self._X[
                                     i].T, 2))
                             )
            probabilities = probabilities / np.sum(probabilities)
            s_i_equals_j.append(self._generator.choice(self._prior_parameters["m"], p=probabilities, size=1))

        s_i_equals_j = np.hstack(s_i_equals_j).astype(int)
        return s_i_equals_j

    def _gamma_block_distribution(self,
                                  most_recent_h: np.ndarray,
                                  most_recent_s_i_equals_j: np.ndarray,
                                  most_recent_h_j: np.ndarray):
        prior_mean = np.zeros(self._prior_parameters["m"] + len(self._X[1, :]))
        prior_mean[-1 * len(self._X[0, :]):] = self._prior_parameters["beta_underbar"]
        prior_precision = block_diag(
            self._prior_parameters["h_mu_underbar"] * most_recent_h * np.eye(self._prior_parameters["m"]),
            self._prior_parameters["H_underbar"])

        Z = np.column_stack(list(most_recent_s_i_equals_j == component for component in range(self._prior_parameters["m"])))

        W_tilde = np.hstack([Z, self._X])
        Q_tilde = np.diag(most_recent_h_j[most_recent_s_i_equals_j])

        posterior_variance = np.linalg.inv(prior_precision + most_recent_h * (W_tilde.T @ Q_tilde @ W_tilde))
        posterior_mean = (posterior_variance @
                          (prior_precision @ prior_mean + most_recent_h * (W_tilde.T @ Q_tilde @ self._y)))

        gamma_draw = self._generator.multivariate_normal(posterior_mean, posterior_variance)
        mu_js = gamma_draw[:self._prior_parameters["m"]]
        betas = gamma_draw[-1 * len(self._X[0, :]):]
        return mu_js, betas

# %%
